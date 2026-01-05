"""
小波变换和频域分析模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DWTForward(nn.Module):
    """可微分小波变换前向"""

    def __init__(self, wave='db4', J=1, mode='zero'):
        super().__init__()
        self.J = J
        self.mode = mode
        self.wave = wave

        # 小波滤波器系数
        if wave == 'haar':
            h0 = torch.tensor([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=torch.float32)
            h1 = torch.tensor([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=torch.float32)
        elif wave == 'db4':
            # Daubechies-4小波系数
            h0 = torch.tensor([
                0.6830127, 1.1830127, 0.3169873, -0.1830127
            ], dtype=torch.float32)
            h1 = torch.tensor([
                -0.1830127, -0.3169873, 1.1830127, -0.6830127
            ], dtype=torch.float32)
        elif wave == 'bior2.2':
            # Biorthogonal 2.2
            h0 = torch.tensor([0, -0.1767767, 0.3535534, 1.0606602,
                               0.3535534, -0.1767767], dtype=torch.float32)
            h1 = torch.tensor([0, 0, 0.3535534, -0.7071068,
                               0.3535534, 0], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported wavelet: {wave}")

        self.register_buffer('h0', h0)
        self.register_buffer('h1', h1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        前向小波变换

        Args:
            x: [B, C, H, W] 输入图像

        Returns:
            LL: [B, C, H/2, W/2] 低频分量
            (LH, HL, HH): 高频分量 each [B, C, H/2, W/2]
        """
        B, C, H, W = x.shape

        # 确保尺寸是偶数
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='reflect')
            H += 1
        if W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0), mode='reflect')
            W += 1

        # 构造卷积核
        filter_len = len(self.h0)
        pad_len = filter_len // 2

        h0_col = self.h0.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        h1_col = self.h1.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        h0_row = self.h0.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        h1_row = self.h1.view(1, 1, 1, -1).repeat(C, 1, 1, 1)

        # 列方向滤波和下采样
        x_pad = F.pad(x, (0, 0, pad_len, pad_len), mode='reflect')
        L = F.conv2d(x_pad, h0_col, groups=C, stride=(2, 1))
        H = F.conv2d(x_pad, h1_col, groups=C, stride=(2, 1))

        # 行方向滤波和下采样
        L_pad = F.pad(L, (pad_len, pad_len, 0, 0), mode='reflect')
        H_pad = F.pad(H, (pad_len, pad_len, 0, 0), mode='reflect')

        LL = F.conv2d(L_pad, h0_row, groups=C, stride=(1, 2))
        LH = F.conv2d(L_pad, h1_row, groups=C, stride=(1, 2))
        HL = F.conv2d(H_pad, h0_row, groups=C, stride=(1, 2))
        HH = F.conv2d(H_pad, h1_row, groups=C, stride=(1, 2))

        return LL, (LH, HL, HH)


class DWTInverse(nn.Module):
    """可微分小波逆变换"""

    def __init__(self, wave='db4'):
        super().__init__()
        self.wave = wave

        # 重构滤波器（分析滤波器的时间反转）
        if wave == 'haar':
            g0 = torch.tensor([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=torch.float32)
            g1 = torch.tensor([-1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=torch.float32)
        elif wave == 'db4':
            g0 = torch.tensor([
                -0.1830127, -0.3169873, 1.1830127, -0.6830127
            ], dtype=torch.float32).flip(0)
            g1 = torch.tensor([
                -0.6830127, 1.1830127, 0.3169873, -0.1830127
            ], dtype=torch.float32).flip(0)
        elif wave == 'bior2.2':
            g0 = torch.tensor([0, 0.3535534, -0.7071068,
                               0.3535534, 0, 0], dtype=torch.float32)
            g1 = torch.tensor([0, -0.1767767, 0.3535534, 1.0606602,
                               0.3535534, -0.1767767], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported wavelet: {wave}")

        self.register_buffer('g0', g0)
        self.register_buffer('g1', g1)

    def forward(self, LL: torch.Tensor, detail: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """小波逆变换重构"""
        LH, HL, HH = detail
        B, C, H, W = LL.shape

        filter_len = len(self.g0)
        pad_len = filter_len - 1

        # 构造转置卷积核
        g0_col = self.g0.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        g1_col = self.g1.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        g0_row = self.g0.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        g1_row = self.g1.view(1, 1, 1, -1).repeat(C, 1, 1, 1)

        # 行方向上采样和滤波
        L = F.conv_transpose2d(LL, g0_row, groups=C, stride=(1, 2),
                               padding=(0, pad_len // 2))
        L += F.conv_transpose2d(LH, g1_row, groups=C, stride=(1, 2),
                                padding=(0, pad_len // 2))

        H = F.conv_transpose2d(HL, g0_row, groups=C, stride=(1, 2),
                               padding=(0, pad_len // 2))
        H += F.conv_transpose2d(HH, g1_row, groups=C, stride=(1, 2),
                                padding=(0, pad_len // 2))

        # 列方向上采样和滤波
        x = F.conv_transpose2d(L, g0_col, groups=C, stride=(2, 1),
                               padding=(pad_len // 2, 0))
        x += F.conv_transpose2d(H, g1_col, groups=C, stride=(2, 1),
                                padding=(pad_len // 2, 0))

        return x


class FrequencyEncoder(nn.Module):
    """频域特征编码器"""

    def __init__(self, in_channels: int, hidden_dim: int, wave='db4'):
        super().__init__()
        self.dwt = DWTForward(wave=wave)
        self.hidden_dim = hidden_dim

        # 各频带编码器
        self.ll_encoder = self._make_band_encoder(in_channels, hidden_dim // 4)
        self.lh_encoder = self._make_band_encoder(in_channels, hidden_dim // 4)
        self.hl_encoder = self._make_band_encoder(in_channels, hidden_dim // 4)
        self.hh_encoder = self._make_band_encoder(in_channels, hidden_dim // 4)

        # 频带能量分析器
        self.energy_analyzer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 4, 1),  # 4个频带的权重
            nn.Sigmoid()
        )

        # 自适应特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        )

    def _make_band_encoder(self, in_ch: int, out_ch: int) -> nn.Module:
        """创建频带编码器"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.GroupNorm(max(1, out_ch // 4), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.GroupNorm(max(1, out_ch // 4), out_ch),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: [B, C, H, W] 输入图像

        Returns:
            freq_features: [B, hidden_dim, H/2, W/2] 频域特征
            band_weights: [B, 4, H/2, W/2] 各频带重要性权重
            freq_info: dict 频域分析信息
        """
        # 小波分解
        LL, (LH, HL, HH) = self.dwt(x)

        # 各频带特征编码
        ll_feat = self.ll_encoder(LL)
        lh_feat = self.lh_encoder(LH)
        hl_feat = self.hl_encoder(HL)
        hh_feat = self.hh_encoder(HH)

        # 拼接频域特征
        freq_features = torch.cat([ll_feat, lh_feat, hl_feat, hh_feat], dim=1)

        # 计算频带重要性权重
        band_weights = self.energy_analyzer(freq_features)

        # 特征融合
        freq_features = self.feature_fusion(freq_features)

        # 频域分析信息
        freq_info = {
            'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH,
            'll_feat': ll_feat, 'lh_feat': lh_feat,
            'hl_feat': hl_feat, 'hh_feat': hh_feat,
            'band_energy': {
                'LL': torch.mean(LL ** 2, dim=(2, 3)),
                'LH': torch.mean(LH ** 2, dim=(2, 3)),
                'HL': torch.mean(HL ** 2, dim=(2, 3)),
                'HH': torch.mean(HH ** 2, dim=(2, 3))
            }
        }

        return freq_features, band_weights, freq_info


def test_wavelets():
    """测试小波变换模块"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试数据
    x = torch.randn(2, 3, 256, 256).to(device)

    # 测试DWT
    dwt = DWTForward(wave='db4').to(device)
    idwt = DWTInverse(wave='db4').to(device)

    print("Testing DWT...")
    LL, (LH, HL, HH) = dwt(x)
    print(f"Input: {x.shape}")
    print(f"LL: {LL.shape}, LH: {LH.shape}, HL: {HL.shape}, HH: {HH.shape}")

    # 测试重构
    x_recon = idwt(LL, (LH, HL, HH))
    print(f"Reconstructed: {x_recon.shape}")
    print(f"Reconstruction error: {torch.mean((x - x_recon) ** 2).item():.6f}")

    # 测试频域编码器
    freq_encoder = FrequencyEncoder(3, 128).to(device)
    freq_feat, band_weights, freq_info = freq_encoder(x)
    print(f"Frequency features: {freq_feat.shape}")
    print(f"Band weights: {band_weights.shape}")


if __name__ == "__main__":
    test_wavelets()