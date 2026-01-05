"""
各向异性分析和引导模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict


class StructureTensorAnalyzer(nn.Module):
    """结构张量分析器"""

    def __init__(self, sigma: float = 1.0, kernel_size: int = 5):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size

        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # 高斯平滑核
        self.gaussian_kernel = self._make_gaussian_kernel(sigma, kernel_size)

    def _make_gaussian_kernel(self, sigma: float, kernel_size: int) -> torch.Tensor:
        """生成高斯平滑核"""
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords -= kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        # 2D高斯核
        kernel = g[:, None] * g[None, :]
        return kernel.view(1, 1, kernel_size, kernel_size)

    def rgb_to_gray(self, x: torch.Tensor) -> torch.Tensor:
        """RGB转灰度"""
        if x.size(1) == 3:
            return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        elif x.size(1) == 1:
            return x
        else:
            # 多通道情况，取平均
            return torch.mean(x, dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        计算结构张量并提取各向异性特征

        Args:
            x: [B, C, H, W] 输入图像

        Returns:
            coherence: [B, 1, H, W] 一致性度量 (0-1)
            orientation: [B, 1, H, W] 主方向 (-π/2 to π/2)
            anisotropy: [B, 1, H, W] 各向异性强度 (0-1)
            struct_info: dict 结构张量信息
        """
        # 转为灰度图
        gray = self.rgb_to_gray(x)

        # 计算梯度
        gray_pad = F.pad(gray, (1, 1, 1, 1), mode='reflect')
        Ix = F.conv2d(gray_pad, self.sobel_x.repeat(1, gray.size(1), 1, 1),
                      groups=gray.size(1))
        Iy = F.conv2d(gray_pad, self.sobel_y.repeat(1, gray.size(1), 1, 1),
                      groups=gray.size(1))

        # 结构张量分量
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # 高斯平滑
        gaussian = self.gaussian_kernel.to(x.device)
        pad_size = self.kernel_size // 2

        Ixx_smooth = F.conv2d(F.pad(Ixx, (pad_size,) * 4, mode='reflect'), gaussian)
        Iyy_smooth = F.conv2d(F.pad(Iyy, (pad_size,) * 4, mode='reflect'), gaussian)
        Ixy_smooth = F.conv2d(F.pad(Ixy, (pad_size,) * 4, mode='reflect'), gaussian)

        # 特征值计算（避免数值不稳定）
        eps = 1e-8
        trace = Ixx_smooth + Iyy_smooth + eps
        det = Ixx_smooth * Iyy_smooth - Ixy_smooth * Ixy_smooth

        discriminant = torch.clamp(trace ** 2 - 4 * det, min=0)
        sqrt_discriminant = torch.sqrt(discriminant)

        lambda1 = 0.5 * (trace + sqrt_discriminant)  # 较大特征值
        lambda2 = 0.5 * (trace - sqrt_discriminant)  # 较小特征值

        # 一致性度量 (coherence)
        coherence = torch.clamp((lambda1 - lambda2) / (lambda1 + lambda2 + eps), 0, 1)

        # 主方向 (orientation)
        orientation = 0.5 * torch.atan2(2 * Ixy_smooth, Ixx_smooth - Iyy_smooth + eps)

        # 各向异性强度
        anisotropy = torch.clamp(1 - lambda2 / (lambda1 + eps), 0, 1)

        # 梯度幅值
        gradient_magnitude = torch.sqrt(Ix ** 2 + Iy ** 2 + eps)

        # 结构信息
        struct_info = {
            'Ix': Ix, 'Iy': Iy,
            'Ixx': Ixx_smooth, 'Iyy': Iyy_smooth, 'Ixy': Ixy_smooth,
            'lambda1': lambda1, 'lambda2': lambda2,
            'gradient_mag': gradient_magnitude,
            'trace': trace, 'det': det
        }

        return coherence, orientation, anisotropy, struct_info


class DirectionalFilter(nn.Module):
    """方向敏感滤波器"""

    def __init__(self, in_channels: int, out_channels: int, num_directions: int = 8):
        super().__init__()
        self.num_directions = num_directions
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 方向敏感卷积核
        self.directional_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // num_directions, 3, 1, 1)
            for _ in range(num_directions)
        ])

        # 方向权重生成器
        self.direction_weights = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),  # coherence + orientation + anisotropy
            nn.GELU(),
            nn.Conv2d(16, num_directions, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor, structure_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 输入特征
            structure_features: [B, 3, H, W] [coherence, orientation, anisotropy]

        Returns:
            directional_features: [B, out_channels, H, W]
        """
        # 计算方向权重
        weights = self.direction_weights(structure_features)  # [B, num_directions, H, W]

        # 方向敏感特征提取
        directional_feats = []
        for i, conv in enumerate(self.directional_convs):
            feat = conv(x)  # [B, out_channels//num_directions, H, W]
            weighted_feat = feat * weights[:, i:i + 1, :, :]
            directional_feats.append(weighted_feat)

        # 拼接所有方向特征
        directional_features = torch.cat(directional_feats, dim=1)

        return directional_features


class AnisotropicGuidanceModule(nn.Module):
    """各向异性引导模块"""

    def __init__(self, in_channels: int, hidden_dim: int, num_directions: int = 8):
        super().__init__()
        self.structure_analyzer = StructureTensorAnalyzer()
        self.num_directions = num_directions

        # 结构特征编码器
        self.structure_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 4, 3, 1, 1),  # coherence + orientation + anisotropy
            nn.GroupNorm(4, hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, 3, 1, 1),
            nn.GroupNorm(4, hidden_dim // 4),
        )

        # 方向敏感滤波器
        self.directional_filter = DirectionalFilter(
            in_channels, hidden_dim // 2, num_directions
        )

        # 边缘增强模块
        self.edge_enhancer = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 8, 3, 1, 1),  # 梯度幅值
            nn.GELU(),
            nn.Conv2d(hidden_dim // 8, hidden_dim // 8, 3, 1, 1),
        )

        # 自适应融合
        self.adaptive_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim // 4 + hidden_dim // 2 + hidden_dim // 8, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [B, C, H, W] 输入图像或特征

        Returns:
            guided_features: [B, hidden_dim, H, W] 引导后的特征
            guidance_info: dict 各向异性引导信息
        """
        # 结构张量分析
        coherence, orientation, anisotropy, struct_info = self.structure_analyzer(x)

        # 结构特征编码
        structure_features = torch.cat([coherence, orientation, anisotropy], dim=1)
        structure_feat = self.structure_encoder(structure_features)

        # 方向敏感特征提取
        directional_feat = self.directional_filter(x, structure_features)

        # 边缘增强特征
        edge_feat = self.edge_enhancer(struct_info['gradient_mag'])

        # 特征融合
        combined_feat = torch.cat([structure_feat, directional_feat, edge_feat], dim=1)
        fused_feat = self.adaptive_fusion(combined_feat)

        # 注意力加权
        attention_weights = self.attention(fused_feat)
        guided_features = fused_feat * attention_weights

        # 引导信息
        guidance_info = {
            'coherence': coherence,
            'orientation': orientation,
            'anisotropy': anisotropy,
            'structure_features': structure_features,
            'directional_features': directional_feat,
            'edge_features': edge_feat,
            'attention_weights': attention_weights,
            'struct_tensor_info': struct_info
        }

        return guided_features, guidance_info


def test_anisotropic():
    """测试各向异性模块"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试数据
    x = torch.randn(2, 3, 256, 256).to(device)

    # 测试结构张量分析器
    print("Testing Structure Tensor Analyzer...")
    struct_analyzer = StructureTensorAnalyzer().to(device)
    coherence, orientation, anisotropy, struct_info = struct_analyzer(x)

    print(f"Input: {x.shape}")
    print(f"Coherence: {coherence.shape}, range: [{coherence.min():.3f}, {coherence.max():.3f}]")
    print(f"Orientation: {orientation.shape}, range: [{orientation.min():.3f}, {orientation.max():.3f}]")
    print(f"Anisotropy: {anisotropy.shape}, range: [{anisotropy.min():.3f}, {anisotropy.max():.3f}]")

    # 测试各向异性引导模块
    print("\nTesting Anisotropic Guidance Module...")
    guidance_module = AnisotropicGuidanceModule(3, 128).to(device)
    guided_feat, guidance_info = guidance_module(x)

    print(f"Guided features: {guided_feat.shape}")
    print(f"Attention weights: {guidance_info['attention_weights'].shape}")


if __name__ == "__main__":
    test_anisotropic()