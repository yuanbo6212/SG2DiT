"""
物理先验工具函数模块
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import cv2


def visualize_priors(ir_image: torch.Tensor, vis_image: torch.Tensor,
                     prior_info: Dict, save_path: Optional[str] = None) -> None:
    """
    可视化物理先验信息

    Args:
        ir_image: [B, C, H, W] 红外图像
        vis_image: [B, C, H, W] 可见光图像
        prior_info: 物理先验信息字典
        save_path: 保存路径（可选）
    """
    try:
        # 转换为numpy格式用于可视化
        def tensor_to_numpy(tensor):
            if tensor.dim() == 4:
                tensor = tensor[0]  # 取第一个batch
            if tensor.dim() == 3:
                tensor = tensor[0]  # 取第一个通道
            return tensor.detach().cpu().numpy()

        ir_np = tensor_to_numpy(ir_image)
        vis_np = tensor_to_numpy(vis_image)

        # 创建子图
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # 原始图像
        axes[0, 0].imshow(ir_np, cmap='gray')
        axes[0, 0].set_title('IR Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(vis_np, cmap='gray')
        axes[0, 1].set_title('VIS Image')
        axes[0, 1].axis('off')

        # 频域先验可视化
        if 'ir_freq_feat' in prior_info:
            ir_freq = tensor_to_numpy(prior_info['ir_freq_feat'])
            vis_freq = tensor_to_numpy(prior_info['vis_freq_feat'])

            axes[0, 2].imshow(ir_freq, cmap='jet')
            axes[0, 2].set_title('IR Frequency Features')
            axes[0, 2].axis('off')

            axes[0, 3].imshow(vis_freq, cmap='jet')
            axes[0, 3].set_title('VIS Frequency Features')
            axes[0, 3].axis('off')

        # 各向异性先验可视化
        if 'ir_guidance_info' in prior_info:
            ir_coherence = tensor_to_numpy(prior_info['ir_guidance_info']['coherence'])
            vis_coherence = tensor_to_numpy(prior_info['vis_guidance_info']['coherence'])

            axes[1, 0].imshow(ir_coherence, cmap='hot')
            axes[1, 0].set_title('IR Coherence')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(vis_coherence, cmap='hot')
            axes[1, 1].set_title('VIS Coherence')
            axes[1, 1].axis('off')

            # 方向信息
            ir_orientation = tensor_to_numpy(prior_info['ir_guidance_info']['orientation'])
            vis_orientation = tensor_to_numpy(prior_info['vis_guidance_info']['orientation'])

            axes[1, 2].imshow(ir_orientation, cmap='hsv')
            axes[1, 2].set_title('IR Orientation')
            axes[1, 2].axis('off')

            axes[1, 3].imshow(vis_orientation, cmap='hsv')
            axes[1, 3].set_title('VIS Orientation')
            axes[1, 3].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Prior visualization saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    except Exception as e:
        print(f"Visualization error: {e}")


def compute_metrics(pred: torch.Tensor, ir: torch.Tensor, vis: torch.Tensor) -> Dict[str, float]:
    """
    计算融合图像的评价指标

    Args:
        pred: [B, C, H, W] 预测融合图像
        ir: [B, C, H, W] 红外图像
        vis: [B, C, H, W] 可见光图像

    Returns:
        metrics: dict 评价指标
    """
    try:
        metrics = {}

        # 转换为单通道灰度图
        def to_grayscale(img):
            if img.size(1) > 1:
                return torch.mean(img, dim=1, keepdim=True)
            return img

        pred_gray = to_grayscale(pred)
        ir_gray = to_grayscale(ir)
        vis_gray = to_grayscale(vis)

        # 1. PSNR
        mse_ir = F.mse_loss(pred_gray, ir_gray)
        mse_vis = F.mse_loss(pred_gray, vis_gray)
        metrics['PSNR_IR'] = 20 * torch.log10(1.0 / torch.sqrt(mse_ir + 1e-8)).item()
        metrics['PSNR_VIS'] = 20 * torch.log10(1.0 / torch.sqrt(mse_vis + 1e-8)).item()

        # 2. SSIM（简化版本）
        def ssim_simple(img1, img2):
            mu1 = F.avg_pool2d(img1, 3, 1, 1)
            mu2 = F.avg_pool2d(img2, 3, 1, 1)
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
            sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
            sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2

            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            return ssim_map.mean()

        metrics['SSIM_IR'] = ssim_simple(pred_gray, ir_gray).item()
        metrics['SSIM_VIS'] = ssim_simple(pred_gray, vis_gray).item()

        # 3. 梯度保持度（简化版本）
        def gradient_magnitude(img):
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=torch.float32, device=img.device).view(1, 1, 3, 3)

            grad_x = F.conv2d(F.pad(img, (1, 1, 1, 1), mode='reflect'), sobel_x)
            grad_y = F.conv2d(F.pad(img, (1, 1, 1, 1), mode='reflect'), sobel_y)

            return torch.sqrt(grad_x ** 2 + grad_y ** 2)

        pred_grad = gradient_magnitude(pred_gray)
        ir_grad = gradient_magnitude(ir_gray)
        vis_grad = gradient_magnitude(vis_gray)

        # 梯度保持度 = 融合图梯度与源图梯度的相关性
        max_grad = torch.max(ir_grad, vis_grad)
        grad_preservation = F.cosine_similarity(pred_grad.flatten(), max_grad.flatten(), dim=0)
        metrics['Gradient_Preservation'] = grad_preservation.item()

        # 4. 信息熵
        def entropy(img):
            img_np = img.detach().cpu().numpy().flatten()
            img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255).astype(np.uint8)
            hist, _ = np.histogram(img_np, bins=256, range=(0, 256))
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # 移除0值
            return -np.sum(hist * np.log2(hist))

        metrics['Entropy'] = entropy(pred_gray[0, 0])

        return metrics

    except Exception as e:
        print(f"Metrics computation error: {e}")
        return {'error': str(e)}


def analyze_frequency_bands(dwt_coeffs: Tuple) -> Dict[str, torch.Tensor]:
    """
    分析小波频带特性

    Args:
        dwt_coeffs: (LL, (LH, HL, HH)) 小波系数

    Returns:
        analysis: dict 频带分析结果
    """
    try:
        ll, detail = dwt_coeffs
        lh, hl, hh = detail

        analysis = {}

        # 能量分布
        ll_energy = torch.sum(ll ** 2)
        lh_energy = torch.sum(lh ** 2)
        hl_energy = torch.sum(hl ** 2)
        hh_energy = torch.sum(hh ** 2)

        total_energy = ll_energy + lh_energy + hl_energy + hh_energy

        analysis['energy_distribution'] = {
            'LL': (ll_energy / total_energy).item(),
            'LH': (lh_energy / total_energy).item(),
            'HL': (hl_energy / total_energy).item(),
            'HH': (hh_energy / total_energy).item()
        }

        # 方向性分析
        horizontal_energy = lh_energy + hh_energy  # 水平方向
        vertical_energy = hl_energy + hh_energy  # 垂直方向
        diagonal_energy = hh_energy  # 对角方向

        analysis['directional_energy'] = {
            'horizontal': (horizontal_energy / total_energy).item(),
            'vertical': (vertical_energy / total_energy).item(),
            'diagonal': (diagonal_energy / total_energy).item()
        }

        # 细节丰富度
        detail_richness = (lh_energy + hl_energy + hh_energy) / total_energy
        analysis['detail_richness'] = detail_richness.item()

        return analysis

    except Exception as e:
        print(f"Frequency analysis error: {e}")
        return {'error': str(e)}


def visualize_frequency_response(image: torch.Tensor, save_path: Optional[str] = None) -> None:
    """
    可视化图像的频率响应

    Args:
        image: [B, C, H, W] 输入图像
        save_path: 保存路径（可选）
    """
    try:
        # 转换为numpy
        if image.dim() == 4:
            image = image[0, 0]  # 取第一个batch和通道

        img_np = image.detach().cpu().numpy()

        # FFT变换
        f_transform = np.fft.fft2(img_np)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)

        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原图
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 频谱
        axes[1].imshow(magnitude_spectrum, cmap='hot')
        axes[1].set_title('Magnitude Spectrum')
        axes[1].axis('off')

        # 频率分布
        center = np.array(magnitude_spectrum.shape) // 2
        y, x = np.ogrid[:magnitude_spectrum.shape[0], :magnitude_spectrum.shape[1]]
        radius = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

        # 径向平均
        max_radius = int(min(center))
        radial_profile = []
        for r in range(max_radius):
            mask = (radius >= r) & (radius < r + 1)
            radial_profile.append(magnitude_spectrum[mask].mean())

        axes[2].plot(radial_profile)
        axes[2].set_title('Radial Frequency Distribution')
        axes[2].set_xlabel('Frequency (cycles/pixel)')
        axes[2].set_ylabel('Magnitude')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Frequency response saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    except Exception as e:
        print(f"Frequency visualization error: {e}")


# 测试函数
def test_utils():
    """测试工具函数"""
    print("Testing utils functions...")

    # 创建测试数据
    ir = torch.randn(1, 1, 64, 64)
    vis = torch.randn(1, 1, 64, 64)
    pred = 0.5 * ir + 0.5 * vis

    # 测试指标计算
    metrics = compute_metrics(pred, ir, vis)
    print("Metrics:", metrics)

    # 测试频率分析
    try:
        import pywt
        coeffs = pywt.dwt2(ir[0, 0].numpy(), 'db4')
        ll = torch.from_numpy(coeffs[0])
        detail = tuple(torch.from_numpy(d) for d in coeffs[1])

        freq_analysis = analyze_frequency_bands((ll, detail))
        print("Frequency analysis:", freq_analysis)
    except ImportError:
        print("PyWavelets not available, skipping frequency analysis test")

    print("Utils functions test completed!")


if __name__ == "__main__":
    test_utils()