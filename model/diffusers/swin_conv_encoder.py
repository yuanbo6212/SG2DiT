# model/diffusers/swin_conv_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.cnn.bricks import build_norm_layer


# 直接定义需要的基础模块，避免循环导入

def default_conv(in_channels, out_channels, kernel_size=3, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class NonLinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.silu(x)


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, conv, act):
        super().__init__()
        self.block1 = nn.Sequential(
            conv(dim_in, dim_out, 3),
            act()
        )
        self.block2 = nn.Sequential(
            conv(dim_out, dim_out, 3),
            act()
        )

        if dim_in != dim_out:
            self.shortcut = conv(dim_in, dim_out, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.block2(self.block1(x))


class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.heads, -1, h * w), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = out.reshape(b, -1, h, w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# 导入物理先验模块
from model.diffusers.enhanced_UNet_arch import FrequencyEncoder, AnisotropicGuidanceModule


class WindowAttention(nn.Module):
    """轻量级窗口注意力"""

    def __init__(self, dim, window_size=7, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = max(1, min(num_heads, dim // 8))  # 确保head数量合理

        # 确保 dim 能被 num_heads 整除
        while dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1

        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 如果尺寸小于窗口大小，使用全局注意力
        if H < self.window_size or W < self.window_size:
            return self.global_attention(x)

        # 分割窗口
        x_windows = self.window_partition(x, self.window_size)
        B_w, C_w, Wh, Ww = x_windows.shape

        # 确保维度一致
        assert C_w == C, f"Channel mismatch: expected {C}, got {C_w}"

        # QKV
        qkv = self.qkv(x_windows)  # [B_w, 3*C, Wh, Ww]
        qkv = qkv.reshape(B_w, 3, self.num_heads, self.head_dim, Wh * Ww)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B_w, num_heads, Wh*Ww, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)

        out = (attn @ v).transpose(-2, -1).reshape(B_w, C_w, Wh, Ww)
        out = self.proj(out)

        # 重组窗口
        out = self.window_reverse(out, self.window_size, H, W)
        return out

    def global_attention(self, x):
        """当窗口大小不合适时使用全局注意力"""
        B, C, H, W = x.shape

        # 简化的全局注意力
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)

        out = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
        return self.proj(out)

    def window_partition(self, x, window_size):
        B, C, H, W = x.shape

        # 计算需要的padding
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H, W = H + pad_h, W + pad_w

        # 确保能整除
        assert H % window_size == 0 and W % window_size == 0, f"Size mismatch: H={H}, W={W}, window_size={window_size}"

        # 重塑为窗口
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = windows.view(-1, C, window_size, window_size)

        return windows

    def window_reverse(self, windows, window_size, H, W):
        # 计算padding
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        H_pad, W_pad = H + pad_h, W + pad_w

        # 计算窗口数量
        num_windows_h = H_pad // window_size
        num_windows_w = W_pad // window_size
        B = windows.shape[0] // (num_windows_h * num_windows_w)

        # 重塑
        x = windows.view(B, num_windows_h, num_windows_w, -1, window_size, window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, -1, H_pad, W_pad)

        # 移除padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x


class SwinConvBlock(nn.Module):
    """Swin-Conv混合块"""

    def __init__(self, dim, num_heads=8, window_size=7, conv_ratio=0.6):
        super().__init__()
        self.dim = dim
        self.conv_dim = max(1, int(dim * conv_ratio))
        self.attn_dim = dim - self.conv_dim

        # 修复分组卷积的组数计算
        def get_valid_groups(channels, target_groups):
            """获取有效的分组数，确保能被通道数整除"""
            if channels <= target_groups:
                return 1
            # 找到最大的能整除channels的数，且不超过target_groups
            for groups in range(target_groups, 0, -1):
                if channels % groups == 0:
                    return groups
            return 1

        # 卷积分支
        if self.conv_dim > 0:
            conv_groups = get_valid_groups(self.conv_dim, self.conv_dim // 8)
            norm_groups = get_valid_groups(self.conv_dim, 8)

            self.conv_branch = nn.Sequential(
                nn.Conv2d(self.conv_dim, self.conv_dim, 7, 1, 3, groups=conv_groups),
                nn.GroupNorm(norm_groups, self.conv_dim),
                nn.GELU(),
                nn.Conv2d(self.conv_dim, self.conv_dim, 1),
                nn.GroupNorm(norm_groups, self.conv_dim),
                nn.GELU()
            )
        else:
            self.conv_branch = nn.Identity()

        # 注意力分支
        if self.attn_dim > 0:
            attn_norm_groups = get_valid_groups(self.attn_dim, 8)
            self.attn_branch = nn.Sequential(
                WindowAttention(self.attn_dim, window_size, min(num_heads, max(1, self.attn_dim // 8))),
                nn.GroupNorm(attn_norm_groups, self.attn_dim),
                nn.GELU()
            )
        else:
            self.attn_branch = nn.Identity()

        # 特征融合
        fusion_norm_groups = get_valid_groups(dim, 8)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GroupNorm(fusion_norm_groups, dim),
            nn.GELU()
        )

    def forward(self, x):
        identity = x

        # 通道分割
        if self.conv_dim > 0 and self.attn_dim > 0:
            conv_x = x[:, :self.conv_dim]
            attn_x = x[:, self.conv_dim:]

            conv_out = self.conv_branch(conv_x)
            attn_out = self.attn_branch(attn_x)
            combined = torch.cat([conv_out, attn_out], dim=1)
        elif self.conv_dim > 0:
            combined = self.conv_branch(x)
        elif self.attn_dim > 0:
            combined = self.attn_branch(x)
        else:
            combined = x

        # 融合
        out = self.fusion(combined)
        return out + identity


class SwinConvEncoder(nn.Module):
    """Swin-Conv混合编码器"""

    def __init__(self, in_ch=1, out_ch=1, ch=64, ch_mult=[1, 2, 4], embed_dim=4,
                 enable_freq_prior=True, enable_aniso_prior=False,
                 window_size=7, num_heads=8, **kwargs):
        super().__init__()
        self.depth = len(ch_mult)
        self.enable_freq_prior = enable_freq_prior
        self.enable_aniso_prior = enable_aniso_prior

        # 物理先验模块
        if self.enable_freq_prior:
            self.freq_encoder_ir = FrequencyEncoder(in_ch, ch, wave='db4')
            self.freq_encoder_vis = FrequencyEncoder(in_ch, ch, wave='db4')

        if self.enable_aniso_prior:
            self.aniso_guidance_ir = AnisotropicGuidanceModule(in_ch, ch // 2)
            self.aniso_guidance_vis = AnisotropicGuidanceModule(in_ch, ch // 2)

        # 初始卷积
        self.init_conv1 = default_conv(in_ch, ch, 3)
        self.init_conv2 = default_conv(in_ch, ch, 3)

        # 先验融合层
        if self.enable_freq_prior or self.enable_aniso_prior:
            total_prior_dim = 0
            if self.enable_freq_prior:
                total_prior_dim += ch
            if self.enable_aniso_prior:
                total_prior_dim += ch // 2

            fusion_input_dim = ch + total_prior_dim
            self.prior_fusion = nn.Sequential(
                nn.Conv2d(fusion_input_dim, ch, 3, 1, 1),
                build_norm_layer(dict(type='BN'), ch)[1],
                nn.LeakyReLU(0.2, True),
            )

        # Swin-Conv编码器层
        self.encoder1 = nn.ModuleList()
        self.encoder2 = nn.ModuleList()

        ch_mult = [1] + ch_mult
        for i in range(self.depth):
            dim_in = ch * ch_mult[i]
            dim_out = ch * ch_mult[i + 1]

            # 编码器块
            blocks1 = nn.ModuleList([
                SwinConvBlock(dim_in, num_heads, window_size, conv_ratio=0.7),
                SwinConvBlock(dim_in, num_heads, window_size, conv_ratio=0.7),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if i == (self.depth - 1) else Identity(),
                Downsample(dim_in, dim_out) if i != (self.depth - 1) else default_conv(dim_in, dim_out, 3)
            ])

            blocks2 = nn.ModuleList([
                SwinConvBlock(dim_in, num_heads, window_size, conv_ratio=0.5),
                SwinConvBlock(dim_in, num_heads, window_size, conv_ratio=0.4),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if i == (self.depth - 1) else Identity(),
                Downsample(dim_in, dim_out) if i != (self.depth - 1) else default_conv(dim_in, dim_out)
            ])

            self.encoder1.append(blocks1)
            self.encoder2.append(blocks2)

        # 潜在空间卷积
        mid_dim = ch * ch_mult[-1]
        default_act = NonLinearity
        self.latent_conv1 = ResBlock(dim_in=mid_dim, dim_out=embed_dim, conv=default_conv, act=default_act)
        self.latent_conv2 = ResBlock(dim_in=mid_dim, dim_out=embed_dim, conv=default_conv, act=default_act)

        # 融合模块
        self.conv_fuse = nn.ModuleList()
        self.num_channels = [ch * ch_mult[0], ch * ch_mult[0], ch * ch_mult[0], ch * ch_mult[1],
                             ch * ch_mult[1], ch * ch_mult[2], ch * ch_mult[2], ch * ch_mult[3], ch * ch_mult[3]]

        for i in range(len(self.num_channels)):
            fuse_layers = [
                nn.Conv2d(self.num_channels[i] * 2, self.num_channels[i], kernel_size=3, padding=1),
                build_norm_layer(dict(type='BN'), self.num_channels[i])[1],
                nn.LeakyReLU(0.2, True),
            ]

            if i < 4 and self.enable_freq_prior:
                fuse_layers.extend([
                    nn.Conv2d(self.num_channels[i], self.num_channels[i], 1),
                    nn.Sigmoid()
                ])

            self.conv_fuse.append(nn.Sequential(*fuse_layers))

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x1, x2):
        self.H, self.W = x1.shape[2:]
        x1 = self.check_image_size(x1, self.H, self.W)
        x2 = self.check_image_size(x2, self.H, self.W)

        # 物理先验提取
        prior_info = {}
        prior_feats_ir = []
        prior_feats_vis = []

        if self.enable_freq_prior:
            ir_freq_feat, ir_band_weights, ir_freq_info = self.freq_encoder_ir(x1)
            vis_freq_feat, vis_band_weights, vis_freq_info = self.freq_encoder_vis(x2)

            prior_info.update({
                'ir_freq_feat': ir_freq_feat,
                'vis_freq_feat': vis_freq_feat,
                'ir_band_weights': ir_band_weights,
                'vis_band_weights': vis_band_weights,
                'ir_freq_info': ir_freq_info,
                'vis_freq_info': vis_freq_info
            })

            prior_feats_ir.append(ir_freq_feat)
            prior_feats_vis.append(vis_freq_feat)

        if self.enable_aniso_prior:
            ir_aniso_feat, ir_guidance_info = self.aniso_guidance_ir(x1)
            vis_aniso_feat, vis_guidance_info = self.aniso_guidance_vis(x2)

            prior_info.update({
                'ir_aniso_feat': ir_aniso_feat,
                'vis_aniso_feat': vis_aniso_feat,
                'ir_guidance_info': ir_guidance_info,
                'vis_guidance_info': vis_guidance_info
            })

            prior_feats_ir.append(ir_aniso_feat)
            prior_feats_vis.append(vis_aniso_feat)

        # 初始特征提取
        x1_init = self.init_conv1(x1)
        x2_init = self.init_conv2(x2)

        # 先验融合
        if hasattr(self, 'prior_fusion') and (prior_feats_ir or prior_feats_vis):
            target_size = x1_init.shape[-2:]

            # 处理IR先验
            if prior_feats_ir:
                ir_priors_resized = []
                for feat in prior_feats_ir:
                    if feat.shape[-2:] != target_size:
                        feat_resized = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                    else:
                        feat_resized = feat
                    ir_priors_resized.append(feat_resized)

                x1_combined = torch.cat([x1_init] + ir_priors_resized, dim=1)
                x1 = self.prior_fusion(x1_combined)
            else:
                x1 = x1_init

            # 处理VIS先验
            if prior_feats_vis:
                vis_priors_resized = []
                for feat in prior_feats_vis:
                    if feat.shape[-2:] != target_size:
                        feat_resized = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                    else:
                        feat_resized = feat
                    vis_priors_resized.append(feat_resized)

                x2_combined = torch.cat([x2_init] + vis_priors_resized, dim=1)
                x2 = self.prior_fusion(x2_combined)
            else:
                x2 = x2_init
        else:
            x1, x2 = x1_init, x2_init

        # Swin-Conv编码
        h1 = [x1]
        h2 = [x2]

        # 第一个编码器（IR）
        for i, (b1, b2, attn, downsample) in enumerate(self.encoder1):
            x1 = b1(x1)
            h1.append(x1)
            x1 = b2(x1)
            x1 = attn(x1)
            h1.append(x1)
            x1 = downsample(x1)

        x1 = self.latent_conv1(x1)

        # 第二个编码器（VIS）
        for i, (b1, b2, attn, downsample) in enumerate(self.encoder2):
            x2 = b1(x2)
            h2.append(x2)
            x2 = b2(x2)
            x2 = attn(x2)
            h2.append(x2)
            x2 = downsample(x2)

        x2 = self.latent_conv2(x2)

        # 特征融合
        h = []
        for i in range(len(h1)):
            A = torch.cat((h1[i], h2[i]), dim=1)

            if i < 4 and self.enable_freq_prior and 'ir_band_weights' in prior_info:
                fused = self.conv_fuse[i](A)
                if len(self.conv_fuse[i]) > 3:
                    weight = fused
                    ir_weight = prior_info['ir_band_weights'].mean(dim=1, keepdim=True)
                    ir_weight = F.interpolate(ir_weight, size=h1[i].shape[-2:], mode='bilinear', align_corners=False)
                    adaptive_weight = 0.7 * weight + 0.3 * ir_weight
                    x = adaptive_weight * h1[i] + (1 - adaptive_weight) * h2[i]
                else:
                    x = fused
            else:
                x = self.conv_fuse[i](A)

            h.append(x)

        return x1, x2, h, prior_info