import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Tuple, List


class DWTForward(nn.Module):
    """可微分小波变换前向"""

    def __init__(self, wave='haar', J=1, mode='zero'):
        super().__init__()
        self.J = J
        self.mode = mode
        self.wave = wave

        # Haar小波滤波器
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

        # 注册为buffer，不参与梯度更新
        self.register_buffer('h0', h0)
        self.register_buffer('h1', h1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            LL: [B, C, H/2, W/2] 低频分量
            (LH, HL, HH): 高频分量 each [B, C, H/2, W/2]
        """
        B, C, H, W = x.shape

        # 构造卷积核
        h0_col = self.h0.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        h1_col = self.h1.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        h0_row = self.h0.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        h1_row = self.h1.view(1, 1, 1, -1).repeat(C, 1, 1, 1)

        # 列方向滤波
        x_pad = F.pad(x, (0, 0, 1, 1), mode='reflect')
        L = F.conv2d(x_pad, h0_col, groups=C, stride=(2, 1))
        H = F.conv2d(x_pad, h1_col, groups=C, stride=(2, 1))

        # 行方向滤波
        L_pad = F.pad(L, (1, 1, 0, 0), mode='reflect')
        H_pad = F.pad(H, (1, 1, 0, 0), mode='reflect')

        LL = F.conv2d(L_pad, h0_row, groups=C, stride=(1, 2))
        LH = F.conv2d(L_pad, h1_row, groups=C, stride=(1, 2))
        HL = F.conv2d(H_pad, h0_row, groups=C, stride=(1, 2))
        HH = F.conv2d(H_pad, h1_row, groups=C, stride=(1, 2))

        return LL, (LH, HL, HH)


class DWTInverse(nn.Module):
    """可微分小波逆变换"""

    def __init__(self, wave='haar'):
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

        self.register_buffer('g0', g0)
        self.register_buffer('g1', g1)

    def forward(self, LL: Tensor, detail: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """重构图像"""
        LH, HL, HH = detail
        B, C, H, W = LL.shape

        # 构造转置卷积核
        g0_col = self.g0.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        g1_col = self.g1.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        g0_row = self.g0.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        g1_row = self.g1.view(1, 1, 1, -1).repeat(C, 1, 1, 1)

        # 行方向上采样和滤波
        L = F.conv_transpose2d(LL, g0_row, groups=C, stride=(1, 2),
                               padding=(0, self.g0.size(0) // 2))
        L += F.conv_transpose2d(LH, g1_row, groups=C, stride=(1, 2),
                                padding=(0, self.g1.size(0) // 2))

        H = F.conv_transpose2d(HL, g0_row, groups=C, stride=(1, 2),
                               padding=(0, self.g0.size(0) // 2))
        H += F.conv_transpose2d(HH, g1_row, groups=C, stride=(1, 2),
                                padding=(0, self.g1.size(0) // 2))

        # 列方向上采样和滤波
        x = F.conv_transpose2d(L, g0_col, groups=C, stride=(2, 1),
                               padding=(self.g0.size(0) // 2, 0))
        x += F.conv_transpose2d(H, g1_col, groups=C, stride=(2, 1),
                                padding=(self.g1.size(0) // 2, 0))

        return x


class FrequencyEncoder(nn.Module):
    """频域特征编码器"""

    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.dwt = DWTForward(wave='db4')

        # 各频带编码器
        self.ll_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 4, 3, 1, 1),
            nn.GroupNorm(4, hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, 3, 1, 1),
        )

        self.lh_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 4, 3, 1, 1),
            nn.GroupNorm(4, hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, 3, 1, 1),
        )

        self.hl_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 4, 3, 1, 1),
            nn.GroupNorm(4, hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, 3, 1, 1),
        )

        self.hh_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 4, 3, 1, 1),
            nn.GroupNorm(4, hidden_dim // 4),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, 3, 1, 1),
        )

        # 频带能量分析器
        self.energy_analyzer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 4, 1),  # 4个频带的权重
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            freq_features: [B, hidden_dim, H/2, W/2] 频域特征
            band_weights: [B, 4, H/2, W/2] 各频带重要性权重
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

        return freq_features, band_weights