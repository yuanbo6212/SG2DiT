"""
物理先验编码器主模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from .wavelets import FrequencyEncoder
from .anisotropic import AnisotropicGuidanceModule


class SpatialEncoder(nn.Module):
    """空域特征编码器"""

    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # 第一层
            nn.Conv2d(in_channels, hidden_dim // 4, 3, 1, 1),
            nn.GroupNorm(4, hidden_dim // 4),
            nn.GELU(),

            # 第二层（下采样）
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 3, 2, 1),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.GELU(),

            # 第三层
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, 1, 1),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.GELU(),
        )

        # 残差连接
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 1, 2, 0),
            nn.GroupNorm(4, hidden_dim // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            features: [B, hidden_dim//2, H/2, W/2]
        """
        feat = self.conv_blocks(x)
        res = self.residual(x)
        return feat + res


class CrossModalFusion(nn.Module):
    """跨模态融合模块"""

    def __init__(self, hidden_dim: int):
        super().__init__()

        # 模态特异性编码器
        self.ir_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, 1, 1),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.GELU(),
        )

        self.vis_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, 1, 1),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.GELU(),
        )

        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim // 2, num_heads=4, batch_first=True
        )

        # 门控融合
        self.gate_generator = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, 1, 1),
            nn.Sigmoid()
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )

    def forward(self, ir_feat: torch.Tensor, vis_feat: torch.Tensor) -> torch.Tensor:
        """
        跨模态特征融合

        Args:
            ir_feat: [B, hidden_dim, H, W] IR特征
            vis_feat: [B, hidden_dim, H, W] VIS特征

        Returns:
            fused_feat: [B, hidden_dim, H, W] 融合特征
        """
        B, C, H, W = ir_feat.shape

        # 模态特异性特征
        ir_specific = self.ir_branch(ir_feat)
        vis_specific = self.vis_branch(vis_feat)

        # 重塑为序列形式进行交叉注意力
        ir_seq = ir_specific.flatten(2).transpose(1, 2)  # [B, HW, C//2]
        vis_seq = vis_specific.flatten(2).transpose(1, 2)  # [B, HW, C//2]

        # 交叉注意力：IR查询VIS
        ir_attended, _ = self.cross_attention(ir_seq, vis_seq, vis_seq)
        ir_attended = ir_attended.transpose(1, 2).reshape(B, C // 2, H, W)

        # 交叉注意力：VIS查询IR
        vis_attended, _ = self.cross_attention(vis_seq, ir_seq, ir_seq)
        vis_attended = vis_attended.transpose(1, 2).reshape(B, C // 2, H, W)

        # 拼接交叉注意力特征
        cross_feat = torch.cat([ir_attended, vis_attended], dim=1)

        # 门控融合
        gate = self.gate_generator(cross_feat)
        gated_feat = gate * ir_feat + (1 - gate) * vis_feat

        # 最终融合
        combined = torch.cat([cross_feat, gated_feat], dim=1)
        fused_feat = self.fusion_conv(combined)

        return fused_feat


class PhysicalPriorEncoder(nn.Module):
    """物理先验增强编码器"""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 128, wave: str = 'db4'):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 频域编码器
        self.freq_encoder = FrequencyEncoder(in_channels, hidden_dim, wave)

        # 各向异性引导模块
        self.aniso_guidance = AnisotropicGuidanceModule(in_channels, hidden_dim)

        # 空域编码器
        self.spatial_encoder = SpatialEncoder(in_channels, hidden_dim)

        # 跨模态融合
        self.cross_modal_fusion = CrossModalFusion(hidden_dim)

        # 多尺度特征融合
        self.multiscale_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim * 2, 1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )

        # 自适应权重生成器
        self.adaptive_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim * 3, hidden_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, 3, 1),  # freq, aniso, spatial权重
            nn.Softmax(dim=1)
        )

        # 最终特征细化
        self.feature_refiner = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )

    def forward(self, ir_img: torch.Tensor, vis_img: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        物理先验编码器前向传播

        Args:
            ir_img: [B, C, H, W] 红外图像
            vis_img: [B, C, H, W] 可见光图像

        Returns:
            enhanced_features: [B, hidden_dim, H/2, W/2] 增强特征
            prior_info: dict 物理先验信息
        """
        B, C, H, W = ir_img.shape

        # === 频域先验提取 ===
        ir_freq_feat, ir_band_weights, ir_freq_info = self.freq_encoder(ir_img)
        vis_freq_feat, vis_band_weights, vis_freq_info = self.freq_encoder(vis_img)

        # 频域特征融合
        freq_feat = self.cross_modal_fusion(ir_freq_feat, vis_freq_feat)

        # === 各向异性先验提取 ===
        ir_aniso_feat, ir_guidance_info = self.aniso_guidance(ir_img)
        vis_aniso_feat, vis_guidance_info = self.aniso_guidance(vis_img)

        # 各向异性特征融合
        aniso_feat = self.cross_modal_fusion(ir_aniso_feat, vis_aniso_feat)

        # === 空域特征提取 ===
        ir_spatial_feat = self.spatial_encoder(ir_img)
        vis_spatial_feat = self.spatial_encoder(vis_img)

        # 空域特征融合
        spatial_feat = self.cross_modal_fusion(ir_spatial_feat, vis_spatial_feat)

        # === 多模态自适应融合 ===
        # 拼接所有特征用于权重计算
        all_features = torch.cat([freq_feat, aniso_feat, spatial_feat], dim=1)

        # 计算自适应权重
        weights = self.adaptive_weights(all_features)  # [B, 3, 1, 1]

        # 加权融合
        weighted_freq = weights[:, 0:1, :, :] * freq_feat
        weighted_aniso = weights[:, 1:2, :, :] * aniso_feat
        weighted_spatial = weights[:, 2:3, :, :] * spatial_feat

        # 多尺度特征融合
        enhanced_features = self.multiscale_fusion(all_features)

        # 残差连接
        weighted_combined = weighted_freq + weighted_aniso + weighted_spatial
        enhanced_features = enhanced_features + weighted_combined

        # 特征细化
        enhanced_features = self.feature_refiner(enhanced_features)

        # 收集先验信息
        prior_info = {
            # 频域信息
            'freq_features': freq_feat,
            'ir_freq_info': ir_freq_info,
            'vis_freq_info': vis_freq_info,
            'ir_band_weights': ir_band_weights,
            'vis_band_weights': vis_band_weights,

            # 各向异性信息
            'aniso_features': aniso_feat,
            'ir_guidance_info': ir_guidance_info,
            'vis_guidance_info': vis_guidance_info,

            # 空域信息
            'spatial_features': spatial_feat,

            # 融合权重
            'adaptive_weights': weights,
            'enhanced_features': enhanced_features
        }

        return enhanced_features, prior_info


def test_encoder():
    """测试物理先验编码器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试数据
    B, H, W = 2, 256, 256
    ir_img = torch.randn(B, 1, H, W).to(device)
    vis_img = torch.randn(B, 1, H, W).to(device)

    print("Testing Physical Prior Encoder...")

    # 初始化编码器
    encoder = PhysicalPriorEncoder(in_channels=1, hidden_dim=128).to(device)

    # 前向传播
    with torch.no_grad():
        enhanced_features, prior_info = encoder(ir_img, vis_img)

        print(f"Input IR: {ir_img.shape}")
        print(f"Input VIS: {vis_img.shape}")
        print(f"Enhanced features: {enhanced_features.shape}")

        print("\nPrior information:")
        print(f"- Frequency features: {prior_info['freq_features'].shape}")
        print(f"- Anisotropic features: {prior_info['aniso_features'].shape}")
        print(f"- Spatial features: {prior_info['spatial_features'].shape}")
        print(f"- Adaptive weights: {prior_info['adaptive_weights'].shape}")

        # 统计信息
        print(f"\nFeature statistics:")
        print(f"- Enhanced features range: [{enhanced_features.min():.3f}, {enhanced_features.max():.3f}]")
        print(f"- Adaptive weights: {prior_info['adaptive_weights'].squeeze()}")


if __name__ == "__main__":
    test_encoder()