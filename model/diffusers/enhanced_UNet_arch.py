import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.cnn.bricks import build_norm_layer
import functools
from model.diffusers.module_util import (
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock,
    LinearAttention,
    PreNorm, Residual, Identity)
from model.head.mods import HFRM

# 导入物理先验模块
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from physical_priors import FrequencyEncoder, AnisotropicGuidanceModule


class FrequencyAwareResBlock(nn.Module):
    """融入频域信息的残差块"""

    def __init__(self, dim_in, dim_out, freq_dim=None, conv=default_conv, act=NonLinearity()):
        super().__init__()
        self.conv = conv
        self.act = act  # 确保act不为None

        # 原始残差块 - 明确传递激活函数
        self.res_block = ResBlock(dim_in=dim_in, dim_out=dim_out, conv=conv, act=act)

        # 频域调制模块（可选）
        if freq_dim is not None:
            self.freq_modulation = nn.Sequential(
                nn.Conv2d(freq_dim, dim_out, 1),
                nn.Sigmoid()
            )
        else:
            self.freq_modulation = None

    def forward(self, x, freq_feat=None):
        """
        Args:
            x: [B, dim_in, H, W] 输入特征
            freq_feat: [B, freq_dim, H, W] 频域特征（可选）
        """
        out = self.res_block(x)

        # 频域调制
        if self.freq_modulation is not None and freq_feat is not None:
            # 调整频域特征尺寸
            if freq_feat.shape[-2:] != out.shape[-2:]:
                freq_feat = F.interpolate(freq_feat, size=out.shape[-2:], mode='bilinear', align_corners=False)

            freq_gate = self.freq_modulation(freq_feat)
            out = out * freq_gate

        return out


class EnhancedEncode(nn.Module):
    """增强版编码器，集成频域和各向异性先验"""

    def __init__(self, in_ch=1, out_ch=1, ch=64, ch_mult=[1, 2, 4, 4], embed_dim=4,
                 enable_freq_prior=True, enable_aniso_prior=True):
        super().__init__()
        self.depth = len(ch_mult)
        self.enable_freq_prior = enable_freq_prior
        self.enable_aniso_prior = enable_aniso_prior

        # === 物理先验模块 ===
        if self.enable_freq_prior:
            self.freq_encoder_ir = FrequencyEncoder(in_ch, ch, wave='db4')
            self.freq_encoder_vis = FrequencyEncoder(in_ch, ch, wave='db4')

        if self.enable_aniso_prior:
            self.aniso_guidance_ir = AnisotropicGuidanceModule(in_ch, ch // 2)
            self.aniso_guidance_vis = AnisotropicGuidanceModule(in_ch, ch // 2)

        # === 原有编码器结构（修改为频域感知） ===
        # 初始卷积层
        self.init_conv1 = default_conv(in_ch, ch, 3)
        self.init_conv2 = default_conv(in_ch, ch, 3)

        # === 修复：正确计算先验融合的输入通道数 ===
        if self.enable_freq_prior or self.enable_aniso_prior:
            # 计算总的先验特征维度
            total_prior_dim = 0
            if self.enable_freq_prior:
                total_prior_dim += ch  # 频域特征维度
            if self.enable_aniso_prior:
                total_prior_dim += ch // 2  # 各向异性特征维度

            # prior_fusion的输入维度 = 初始特征(ch) + 先验特征(total_prior_dim)
            fusion_input_dim = ch + total_prior_dim

            self.prior_fusion = nn.Sequential(
                nn.Conv2d(fusion_input_dim, ch, 3, 1, 1),
                build_norm_layer(dict(type='BN'), ch)[1],
                nn.LeakyReLU(0.2, True),
            )

        # === 修复：确保激活函数不为None ===
        default_act = NonLinearity()

        # 编码器层（使用频域感知残差块）
        self.encoder1 = nn.ModuleList([])
        self.encoder2 = nn.ModuleList([])

        ch_mult = [1] + ch_mult
        for i in range(self.depth):
            dim_in = ch * ch_mult[i]
            dim_out = ch * ch_mult[i + 1]

            # 频域调制维度（在较浅层使用）
            freq_mod_dim = ch if (i < 2 and self.enable_freq_prior) else None

            self.encoder1.append(nn.ModuleList([
                FrequencyAwareResBlock(dim_in=dim_in, dim_out=dim_in, freq_dim=freq_mod_dim,
                                       conv=default_conv, act=default_act),
                FrequencyAwareResBlock(dim_in=dim_in, dim_out=dim_in, freq_dim=freq_mod_dim,
                                       conv=default_conv, act=default_act),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if i == (self.depth - 1) else Identity(),
                Downsample(dim_in, dim_out) if i != (self.depth - 1) else default_conv(dim_in, dim_out)
            ]))

            self.encoder2.append(nn.ModuleList([
                FrequencyAwareResBlock(dim_in=dim_in, dim_out=dim_in, freq_dim=freq_mod_dim,
                                       conv=default_conv, act=default_act),
                FrequencyAwareResBlock(dim_in=dim_in, dim_out=dim_in, freq_dim=freq_mod_dim,
                                       conv=default_conv, act=default_act),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if i == (self.depth - 1) else Identity(),
                Downsample(dim_in, dim_out) if i != (self.depth - 1) else default_conv(dim_in, dim_out)
            ]))

        mid_dim = ch * ch_mult[-1]

        self.latent_conv1 = ResBlock(dim_in=mid_dim, dim_out=embed_dim, conv=default_conv, act=default_act)
        self.latent_conv2 = ResBlock(dim_in=mid_dim, dim_out=embed_dim, conv=default_conv, act=default_act)

        # === 增强的特征融合（考虑频域权重） ===
        self.conv_fuse = nn.ModuleList()
        self.num_channels = [ch * ch_mult[0], ch * ch_mult[0], ch * ch_mult[0], ch * ch_mult[1],
                             ch * ch_mult[1], ch * ch_mult[2], ch * ch_mult[2], ch * ch_mult[3], ch * ch_mult[3]]

        for i in range(len(self.num_channels)):
            # 原有融合 + 频域引导的自适应权重
            fuse_layers = [
                nn.Conv2d(self.num_channels[i] * 2, self.num_channels[i], kernel_size=3, padding=1),
                build_norm_layer(dict(type='BN'), self.num_channels[i])[1],
                nn.LeakyReLU(0.2, True),
            ]

            # 添加频域自适应权重生成器（仅在浅层）
            if i < 4 and self.enable_freq_prior:  # 前4层
                fuse_layers.extend([
                    nn.Conv2d(self.num_channels[i], self.num_channels[i], 1),
                    nn.Sigmoid()  # 生成0-1权重
                ])

            self.conv_fuse.append(nn.Sequential(*fuse_layers))

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x1, x2):
        """
        Args:
            x1: [B, C, H, W] IR图像
            x2: [B, C, H, W] VIS图像

        Returns:
            latent1, latent2: 潜在表示
            h: 跳跃连接特征列表
            prior_info: 物理先验信息（用于损失计算）
        """
        self.H, self.W = x1.shape[2:]
        x1 = self.check_image_size(x1, self.H, self.W)
        x2 = self.check_image_size(x2, self.H, self.W)

        # === 物理先验提取 ===
        prior_info = {}
        prior_feats_ir = []
        prior_feats_vis = []

        # 频域先验
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

        # 各向异性先验
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

        # === 初始特征提取 ===
        x1_init = self.init_conv1(x1)
        x2_init = self.init_conv2(x2)

        # === 修复：正确融合物理先验到初始特征 ===
        if hasattr(self, 'prior_fusion') and (prior_feats_ir or prior_feats_vis):
            # 调整所有先验特征到相同尺寸
            target_size = x1_init.shape[-2:]

            # 处理IR分支的先验特征
            ir_priors_resized = []
            for feat in prior_feats_ir:
                if feat.shape[-2:] != target_size:
                    feat_resized = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                else:
                    feat_resized = feat
                ir_priors_resized.append(feat_resized)

            # 处理VIS分支的先验特征
            vis_priors_resized = []
            for feat in prior_feats_vis:
                if feat.shape[-2:] != target_size:
                    feat_resized = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                else:
                    feat_resized = feat
                vis_priors_resized.append(feat_resized)

            # 融合IR分支：初始特征 + 所有先验特征
            if ir_priors_resized:
                x1_combined = torch.cat([x1_init] + ir_priors_resized, dim=1)
                x1 = self.prior_fusion(x1_combined)
            else:
                x1 = x1_init

            # 融合VIS分支：初始特征 + 所有先验特征
            if vis_priors_resized:
                x2_combined = torch.cat([x2_init] + vis_priors_resized, dim=1)
                x2 = self.prior_fusion(x2_combined)
            else:
                x2 = x2_init
        else:
            x1, x2 = x1_init, x2_init

        # === 编码过程 ===
        h1 = [x1]
        h2 = [x2]
        h = []

        # 准备频域调制特征（如果启用）
        freq_mod_feat = None
        if self.enable_freq_prior and 'ir_freq_feat' in prior_info:
            # 使用融合的频域特征进行调制
            freq_mod_feat = 0.5 * (prior_info['ir_freq_feat'] + prior_info['vis_freq_feat'])

        # IR编码器
        for i, (b1, b2, attn, downsample) in enumerate(self.encoder1):
            # 频域调制特征（仅在浅层使用）
            current_freq_feat = freq_mod_feat if (i < 2 and freq_mod_feat is not None) else None

            x1 = b1(x1, current_freq_feat)
            h1.append(x1)
            x1 = b2(x1, current_freq_feat)
            x1 = attn(x1)
            h1.append(x1)
            x1 = downsample(x1)

        x1 = self.latent_conv1(x1)

        # VIS编码器
        for i, (b1, b2, attn, downsample) in enumerate(self.encoder2):
            # 频域调制特征（仅在浅层使用）
            current_freq_feat = freq_mod_feat if (i < 2 and freq_mod_feat is not None) else None

            x2 = b1(x2, current_freq_feat)
            h2.append(x2)
            x2 = b2(x2, current_freq_feat)
            x2 = attn(x2)
            h2.append(x2)
            x2 = downsample(x2)

        x2 = self.latent_conv2(x2)

        # === 增强的特征融合 ===
        for i in range(len(h1)):
            # 原有融合
            A = torch.cat((h1[i], h2[i]), dim=1)

            if i < 4 and self.enable_freq_prior and 'ir_band_weights' in prior_info:
                # 频域引导的自适应融合
                fused = self.conv_fuse[i](A)
                # 最后一层是Sigmoid，产生融合权重
                if len(self.conv_fuse[i]) > 3:  # 有Sigmoid层
                    weight = fused
                    # 使用频域权重引导融合
                    ir_weight = prior_info['ir_band_weights'].mean(dim=1, keepdim=True)  # [B, 1, H/2, W/2]
                    ir_weight = F.interpolate(ir_weight, size=h1[i].shape[-2:], mode='bilinear', align_corners=False)

                    adaptive_weight = 0.7 * weight + 0.3 * ir_weight
                    x = adaptive_weight * h1[i] + (1 - adaptive_weight) * h2[i]
                else:
                    x = fused
            else:
                # 原有融合方式
                x = self.conv_fuse[i](A)

            h.append(x)

        return x1, x2, h, prior_info


# 保持解码器不变，但可以选择性接收先验信息
class EnhancedDecode(nn.Module):
    """解码器保持原有结构，可选择性使用先验信息"""

    def __init__(self, in_ch=1, out_ch=1, ch=64, ch_mult=[1, 2, 4, 4], embed_dim=4):
        super().__init__()
        self.depth = len(ch_mult)

        # === 修复：确保激活函数不为None ===
        default_act = NonLinearity()
        block_class = functools.partial(ResBlock, conv=default_conv, act=default_act)

        self.decoder = nn.ModuleList([])
        ch_mult = [1] + ch_mult
        for i in range(self.depth):
            dim_in = ch * ch_mult[i]
            dim_out = ch * ch_mult[i + 1]
            self.decoder.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if i == (self.depth - 1) else Identity(),
                Upsample(dim_out, dim_in) if i != 0 else default_conv(dim_out, dim_in)
            ]))

        mid_dim = ch * ch_mult[-1]
        self.post_latent_conv = block_class(dim_in=embed_dim, dim_out=mid_dim)
        self.final_conv = nn.Conv2d(ch, out_ch, 3, 1, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x, h, prior_info=None):
        """
        Args:
            x: 潜在表示
            h: 跳跃连接特征
            prior_info: 物理先验信息（可选，用于后续扩展）
        """
        x = self.post_latent_conv(x)

        for i, (b1, b2, attn, upsample) in enumerate(self.decoder):
            x = torch.cat([x, h[-(i * 2 + 1)]], dim=1)
            x = b1(x)

            x = torch.cat([x, h[-(i * 2 + 2)]], dim=1)
            x = b2(x)
            x = attn(x)

            x = upsample(x)

        x = self.final_conv(x + h[0])
        x = torch.tanh(x)
        return x


if __name__ == '__main__':
    test_sample1 = torch.randn(1, 1, 256, 256).cuda()
    test_sample2 = torch.randn(1, 1, 256, 256).cuda()

    # 测试增强版编码器
    enhanced_enc = EnhancedEncode(
        in_ch=1, out_ch=1, ch=8, ch_mult=[4, 4, 4, 8], embed_dim=8,
        enable_freq_prior=True, enable_aniso_prior=True
    ).cuda().eval()

    enhanced_dec = EnhancedDecode(
        in_ch=1, out_ch=1, ch=8, ch_mult=[4, 4, 4, 8], embed_dim=8
    ).cuda().eval()

    x1, x2, h, prior_info = enhanced_enc(test_sample1, test_sample2)
    x = enhanced_dec(x2, h, prior_info)

    print("Enhanced UNet:")
    print(f"Output shape: {x.shape}")
    print(f"Prior info keys: {list(prior_info.keys())}")

    total = sum([param.nelement() for param in enhanced_enc.parameters()])
    print("Enhanced Encoder parameters: %.2fM" % (total / 1e6))