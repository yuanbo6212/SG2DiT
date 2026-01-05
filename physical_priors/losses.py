"""
物理先验损失函数模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

from .wavelets import DWTForward
from .anisotropic import StructureTensorAnalyzer


class FrequencyDomainLoss(nn.Module):
    """频域损失函数"""

    def __init__(self, wave: str = 'db4', weights: Dict[str, float] = None):
        super().__init__()
        self.dwt = DWTForward(wave=wave)

        self.weights = weights or {
            'll_l1': 1.0,  # 低频L1损失
            'll_ssim': 0.5,  # 低频SSIM损失
            'hf_l1': 0.8,  # 高频L1损失
            'hf_ssim': 0.3,  # 高频SSIM损失
            'energy_preserve': 0.4  # 能量保持损失
        }

    def ssim_loss(self, pred: torch.Tensor, target: torch.Tensor,
                  window_size: int = 11, size_average: bool = True) -> torch.Tensor:
        """计算SSIM损失"""

        def gaussian_window(window_size: int, sigma: float = 1.5) -> torch.Tensor:
            coords = torch.arange(window_size, dtype=torch.float32)
            coords -= window_size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            return g[:, None] * g[None, :]

        channel = pred.size(1)
        window = gaussian_window(window_size).expand(channel, 1, window_size, window_size)
        window = window.contiguous().to(pred.device)

        mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean([1, 2, 3])

    def energy_preservation_loss(self, pred_bands: Tuple, target_bands: Tuple) -> torch.Tensor:
        """频带能量保持损失"""
        pred_ll, pred_detail = pred_bands
        target_ll, target_detail = target_bands

        # 低频能量
        pred_ll_energy = torch.sum(pred_ll ** 2, dim=[2, 3])
        target_ll_energy = torch.sum(target_ll ** 2, dim=[2, 3])
        ll_energy_loss = F.l1_loss(pred_ll_energy, target_ll_energy)

        # 高频能量
        hf_energy_loss = 0
        for pred_hf, target_hf in zip(pred_detail, target_detail):
            pred_hf_energy = torch.sum(pred_hf ** 2, dim=[2, 3])
            target_hf_energy = torch.sum(target_hf ** 2, dim=[2, 3])
            hf_energy_loss += F.l1_loss(pred_hf_energy, target_hf_energy)

        return ll_energy_loss + hf_energy_loss / 3

    def forward(self, pred: torch.Tensor, ir: torch.Tensor,
                vis: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算频域损失

        Args:
            pred: [B, C, H, W] 预测的融合图像
            ir: [B, C, H, W] 红外图像
            vis: [B, C, H, W] 可见光图像

        Returns:
            losses: dict 各项频域损失
        """
        try:
            # 小波分解
            pred_ll, pred_detail = self.dwt(pred)
            ir_ll, ir_detail = self.dwt(ir)
            vis_ll, vis_detail = self.dwt(vis)

            losses = {}

            # 低频损失（主要保持IR的结构信息）
            losses['ll_l1'] = F.l1_loss(pred_ll, ir_ll)
            losses['ll_ssim'] = self.ssim_loss(pred_ll, ir_ll)

            # 高频损失（主要保持VIS的细节信息）
            hf_l1_loss = 0
            hf_ssim_loss = 0
            for pred_hf, vis_hf in zip(pred_detail, vis_detail):
                hf_l1_loss += F.l1_loss(pred_hf, vis_hf)
                hf_ssim_loss += self.ssim_loss(pred_hf, vis_hf)

            losses['hf_l1'] = hf_l1_loss / 3
            losses['hf_ssim'] = hf_ssim_loss / 3

            # 能量保持损失
            target_ll = 0.7 * ir_ll + 0.3 * vis_ll  # 低频主要来自IR
            target_detail = vis_detail  # 高频主要来自VIS
            losses['energy_preserve'] = self.energy_preservation_loss(
                (pred_ll, pred_detail), (target_ll, target_detail)
            )

            # 加权总损失
            total_loss = sum(self.weights[k] * v for k, v in losses.items()
                             if k in self.weights)
            losses['freq_total'] = total_loss

        except Exception as e:
            # 处理尺寸不匹配等异常
            print(f"Frequency domain loss error: {e}")
            losses = {k: torch.tensor(0.0, device=pred.device) for k in self.weights.keys()}
            losses['freq_total'] = torch.tensor(0.0, device=pred.device)

        return losses


class AnisotropicLoss(nn.Module):
    """各向异性损失函数"""

    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        self.structure_analyzer = StructureTensorAnalyzer()

        self.weights = weights or {
            'coherence': 0.5,  # 一致性损失
            'orientation': 0.7,  # 方向损失
            'anisotropy': 0.4,  # 各向异性损失
            'gradient': 0.6  # 梯度保持损失
        }

    def direction_difference(self, angle1: torch.Tensor, angle2: torch.Tensor) -> torch.Tensor:
        """计算方向角度差异（考虑周期性）"""
        diff = torch.abs(angle1 - angle2)
        diff = torch.min(diff, np.pi - diff)  # 处理角度周期性
        return diff

    def forward(self, pred: torch.Tensor, ir: torch.Tensor, vis: torch.Tensor,
                ir_guidance_info: Dict, vis_guidance_info: Dict) -> Dict[str, torch.Tensor]:
        """
        计算各向异性损失

        Args:
            pred: [B, C, H, W] 预测融合图像
            ir: [B, C, H, W] 红外图像
            vis: [B, C, H, W] 可见光图像
            ir_guidance_info: IR各向异性信息
            vis_guidance_info: VIS各向异性信息

        Returns:
            losses: dict 各项各向异性损失
        """
        # 计算融合图像的结构信息
        pred_coherence, pred_orientation, pred_anisotropy, _ = self.structure_analyzer(pred)

        # 获取IR和VIS的结构信息
        ir_coherence = ir_guidance_info['coherence']
        ir_orientation = ir_guidance_info['orientation']
        ir_anisotropy = ir_guidance_info['anisotropy']

        vis_coherence = vis_guidance_info['coherence']
        vis_orientation = vis_guidance_info['orientation']
        vis_anisotropy = vis_guidance_info['anisotropy']

        losses = {}

        # 一致性损失：融合图应保持较强的一致性
        target_coherence = torch.max(ir_coherence, vis_coherence)
        losses['coherence'] = F.l1_loss(pred_coherence, target_coherence)

        # 方向损失：根据一致性强度选择主导方向
        coherence_weight = ir_coherence / (ir_coherence + vis_coherence + 1e-8)
        target_orientation = coherence_weight * ir_orientation + (1 - coherence_weight) * vis_orientation

        orientation_diff = self.direction_difference(pred_orientation, target_orientation)
        losses['orientation'] = torch.mean(orientation_diff)

        # 各向异性损失：保持较强的各向异性
        target_anisotropy = torch.max(ir_anisotropy, vis_anisotropy)
        losses['anisotropy'] = F.l1_loss(pred_anisotropy, target_anisotropy)

        # 梯度保持损失
        ir_grad_mag = ir_guidance_info['struct_tensor_info']['gradient_mag']
        vis_grad_mag = vis_guidance_info['struct_tensor_info']['gradient_mag']
        pred_grad_mag = self.structure_analyzer(pred)[3]['gradient_mag']

        target_grad_mag = torch.max(ir_grad_mag, vis_grad_mag)
        losses['gradient'] = F.l1_loss(pred_grad_mag, target_grad_mag)

        # 加权总损失
        total_loss = sum(self.weights[k] * v for k, v in losses.items()
                         if k in self.weights)
        losses['aniso_total'] = total_loss

        return losses


class EdgePreservationLoss(nn.Module):
    """边缘保持损失"""

    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()

        self.weights = weights or {
            'sobel': 1.0,  # Sobel边缘损失
            'laplacian': 0.5,  # Laplacian边缘损失
            'canny': 0.3  # Canny边缘保持损失
        }

        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        # Laplacian算子
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3))

    def get_edges_sobel(self, img: torch.Tensor) -> torch.Tensor:
        """使用Sobel算子提取边缘"""
        if img.size(1) > 1:
            img = torch.mean(img, dim=1, keepdim=True)

        grad_x = F.conv2d(F.pad(img, (1, 1, 1, 1), mode='reflect'),
                          self.sobel_x.repeat(1, img.size(1), 1, 1), groups=img.size(1))
        grad_y = F.conv2d(F.pad(img, (1, 1, 1, 1), mode='reflect'),
                          self.sobel_y.repeat(1, img.size(1), 1, 1), groups=img.size(1))

        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    def get_edges_laplacian(self, img: torch.Tensor) -> torch.Tensor:
        """使用Laplacian算子提取边缘"""
        if img.size(1) > 1:
            img = torch.mean(img, dim=1, keepdim=True)

        edges = F.conv2d(F.pad(img, (1, 1, 1, 1), mode='reflect'),
                         self.laplacian.repeat(1, img.size(1), 1, 1), groups=img.size(1))

        return torch.abs(edges)

    def canny_edge_preserve_loss(self, pred: torch.Tensor, ir: torch.Tensor,
                                 vis: torch.Tensor) -> torch.Tensor:
        """Canny风格的边缘保持损失"""
        pred_edges = self.get_edges_sobel(pred)
        ir_edges = self.get_edges_sobel(ir)
        vis_edges = self.get_edges_sobel(vis)

        # 融合图的边缘应该保持两个模态的强边缘
        target_edges = torch.max(ir_edges, vis_edges)

        # 使用平滑L1损失，对强边缘更敏感
        return F.smooth_l1_loss(pred_edges, target_edges)

    def forward(self, pred: torch.Tensor, ir: torch.Tensor,
                vis: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算边缘保持损失

        Args:
            pred: [B, C, H, W] 预测融合图像
            ir: [B, C, H, W] 红外图像
            vis: [B, C, H, W] 可见光图像

        Returns:
            losses: dict 各项边缘损失
        """
        losses = {}

        # Sobel边缘损失
        pred_sobel = self.get_edges_sobel(pred)
        ir_sobel = self.get_edges_sobel(ir)
        vis_sobel = self.get_edges_sobel(vis)
        target_sobel = torch.max(ir_sobel, vis_sobel)
        losses['sobel'] = F.l1_loss(pred_sobel, target_sobel)

        # Laplacian边缘损失
        pred_laplacian = self.get_edges_laplacian(pred)
        ir_laplacian = self.get_edges_laplacian(ir)
        vis_laplacian = self.get_edges_laplacian(vis)
        target_laplacian = torch.max(ir_laplacian, vis_laplacian)
        losses['laplacian'] = F.l1_loss(pred_laplacian, target_laplacian)

        # Canny风格边缘保持损失
        losses['canny'] = self.canny_edge_preserve_loss(pred, ir, vis)

        # 加权总损失
        total_loss = sum(self.weights[k] * v for k, v in losses.items()
                         if k in self.weights)
        losses['edge_total'] = total_loss

        return losses


class ThermalTargetLoss(nn.Module):
    """热目标保持损失"""

    def __init__(self, threshold: float = 0.7, weights: Dict[str, float] = None):
        super().__init__()
        self.threshold = threshold

        self.weights = weights or {
            'intensity': 1.0,  # 强度保持
            'contrast': 0.8,  # 对比度保持
            'saliency': 0.6  # 显著性保持
        }

    def get_thermal_mask(self, ir: torch.Tensor, threshold: float = None) -> torch.Tensor:
        """生成热目标掩膜"""
        if threshold is None:
            threshold = self.threshold

        # 归一化到[0,1]
        ir_norm = (ir - ir.min()) / (ir.max() - ir.min() + 1e-8)

        # 基于阈值的二值掩膜
        binary_mask = (ir_norm > threshold).float()

        # 形态学操作去除噪声
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=ir.device) / (kernel_size ** 2)
        smooth_mask = F.conv2d(F.pad(binary_mask, (1, 1, 1, 1), mode='reflect'), kernel)

        return (smooth_mask > 0.5).float()

    def local_contrast(self, img: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """计算局部对比度"""
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=img.device) / (kernel_size ** 2)

        # 局部均值
        local_mean = F.conv2d(F.pad(img, (kernel_size // 2,) * 4, mode='reflect'),
                              kernel.repeat(img.size(1), 1, 1, 1), groups=img.size(1))

        # 局部方差
        local_var = F.conv2d(F.pad(img ** 2, (kernel_size // 2,) * 4, mode='reflect'),
                             kernel.repeat(img.size(1), 1, 1, 1), groups=img.size(1)) - local_mean ** 2

        # 对比度 = std / mean
        local_contrast = torch.sqrt(torch.clamp(local_var, min=1e-8)) / (local_mean + 1e-8)

        return local_contrast

    def forward(self, pred: torch.Tensor, ir: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算热目标保持损失

        Args:
            pred: [B, C, H, W] 预测融合图像
            ir: [B, C, H, W] 红外图像

        Returns:
            losses: dict 各项热目标损失
        """
        # 生成热目标掩膜
        thermal_mask = self.get_thermal_mask(ir)

        losses = {}

        # 强度保持损失：在热目标区域保持IR的亮度
        masked_pred = pred * thermal_mask
        masked_ir = ir * thermal_mask
        losses['intensity'] = F.l1_loss(masked_pred, masked_ir)

        # 对比度保持损失：保持热目标的局部对比度
        pred_contrast = self.local_contrast(pred)
        ir_contrast = self.local_contrast(ir)
        masked_pred_contrast = pred_contrast * thermal_mask
        masked_ir_contrast = ir_contrast * thermal_mask
        losses['contrast'] = F.l1_loss(masked_pred_contrast, masked_ir_contrast)

        # 显著性保持损失：热目标在融合图中应该突出
        pred_saliency = pred - F.avg_pool2d(pred, kernel_size=7, stride=1, padding=3)
        ir_saliency = ir - F.avg_pool2d(ir, kernel_size=7, stride=1, padding=3)
        masked_pred_saliency = pred_saliency * thermal_mask
        masked_ir_saliency = ir_saliency * thermal_mask
        losses['saliency'] = F.l1_loss(masked_pred_saliency, masked_ir_saliency)

        # 加权总损失
        total_loss = sum(self.weights[k] * v for k, v in losses.items()
                         if k in self.weights)
        losses['thermal_total'] = total_loss

        return losses


class PhysicalPriorLoss(nn.Module):
    """物理先验总损失函数"""

    def __init__(self, loss_weights: Dict[str, float] = None, **kwargs):
        super().__init__()

        # 默认损失权重
        default_weights = {
            'spatial_l1': 1.0,
            'spatial_ssim': 0.5,
            'frequency': 0.8,
            'anisotropic': 0.6,
            'edge_preserve': 0.7,
            'thermal_preserve': 0.5
        }

        self.loss_weights = loss_weights or default_weights

        # 初始化各项损失
        self.freq_loss = FrequencyDomainLoss(**kwargs.get('freq_args', {}))
        self.aniso_loss = AnisotropicLoss(**kwargs.get('aniso_args', {}))
        self.edge_loss = EdgePreservationLoss(**kwargs.get('edge_args', {}))
        self.thermal_loss = ThermalTargetLoss(**kwargs.get('thermal_args', {}))

    def ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """SSIM损失"""
        return self.freq_loss.ssim_loss(pred, target)

    def forward(self, pred: torch.Tensor, ir: torch.Tensor, vis: torch.Tensor,
                prior_info: Dict) -> Dict[str, torch.Tensor]:
        """
        计算物理先验总损失

        Args:
            pred: [B, C, H, W] 预测融合图像
            ir: [B, C, H, W] 红外图像
            vis: [B, C, H, W] 可见光图像
            prior_info: dict 物理先验信息

        Returns:
            losses: dict 所有损失项
        """
        all_losses = {}

        # 基础空域损失
        target_spatial = 0.5 * (ir + vis)  # 简单平均作为基准
        all_losses['spatial_l1'] = F.l1_loss(pred, target_spatial)
        all_losses['spatial_ssim'] = self.ssim_loss(pred, target_spatial)

        # 频域损失
        freq_losses = self.freq_loss(pred, ir, vis)
        all_losses.update({f'freq_{k}': v for k, v in freq_losses.items()})

        # 各向异性损失
        if 'ir_guidance_info' in prior_info and 'vis_guidance_info' in prior_info:
            aniso_losses = self.aniso_loss(pred, ir, vis,
                                           prior_info['ir_guidance_info'],
                                           prior_info['vis_guidance_info'])
            all_losses.update({f'aniso_{k}': v for k, v in aniso_losses.items()})

        # 边缘保持损失
        edge_losses = self.edge_loss(pred, ir, vis)
        all_losses.update({f'edge_{k}': v for k, v in edge_losses.items()})

        # 热目标保持损失
        thermal_losses = self.thermal_loss(pred, ir)
        all_losses.update({f'thermal_{k}': v for k, v in thermal_losses.items()})

        # 计算加权总损失
        total_loss = 0
        for loss_type, weight in self.loss_weights.items():
            if loss_type in ['frequency', 'anisotropic', 'edge_preserve', 'thermal_preserve']:
                # 使用子损失的总和
                key = f'{loss_type.split("_")[0]}_total'
                if key in all_losses:
                    total_loss += weight * all_losses[key]
            else:
                # 直接使用损失值
                if loss_type in all_losses:
                    total_loss += weight * all_losses[loss_type]

        all_losses['total'] = total_loss

        return all_losses


def test_losses():
    """测试损失函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试数据
    B, H, W = 2, 128, 128
    pred = torch.randn(B, 1, H, W).to(device)
    ir = torch.randn(B, 1, H, W).to(device)
    vis = torch.randn(B, 1, H, W).to(device)

    # 模拟先验信息
    from .anisotropic import AnisotropicGuidanceModule
    guidance_module = AnisotropicGuidanceModule(1, 64).to(device)

    with torch.no_grad():
        _, ir_guidance_info = guidance_module(ir)
        _, vis_guidance_info = guidance_module(vis)

        prior_info = {
            'ir_guidance_info': ir_guidance_info,
            'vis_guidance_info': vis_guidance_info
        }

        # 测试损失函数
        print("Testing Physical Prior Loss...")
        loss_fn = PhysicalPriorLoss().to(device)
        losses = loss_fn(pred, ir, vis, prior_info)

        print("Loss components:")
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.item():.4f}")


if __name__ == "__main__":
    test_losses()