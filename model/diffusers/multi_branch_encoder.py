import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.cnn.bricks import build_norm_layer
from model.diffusers.module_util import (
    NonLinearity, Upsample, Downsample, default_conv, ResBlock,
    LinearAttention, PreNorm, Residual, Identity
)

# 导入物理先验模块
try:
    from physical_priors import FrequencyEncoder, AnisotropicGuidanceModule
except ImportError:
    # 如果没有单独的文件，从enhanced_UNet_arch导入
    from model.diffusers.enhanced_UNet_arch import FrequencyEncoder, AnisotropicGuidanceModule


class ModalitySpecificBranch(nn.Module):
    """模态特化分支"""

    def __init__(self, in_ch, out_ch, branch_type='ir'):
        super().__init__()
        self.branch_type = branch_type

        if branch_type == 'ir':
            # IR分支：更关注低频/对比度
            self.blocks = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 5, 1, 2),  # 大核关注低频
                nn.GroupNorm(8, out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                nn.GroupNorm(8, out_ch),
                nn.GELU()
            )
        else:
            # VIS分支：更关注边缘/纹理
            self.blocks = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 1, 1),  # 标准核
                nn.GroupNorm(8, out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1, groups=out_ch // 8),  # 深度卷积关注局部
                nn.GroupNorm(8, out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 1)  # 点卷积
            )

    def forward(self, x):
        return self.blocks(x)


class AdaptiveFusion(nn.Module):
    """自适应融合模块"""

    def __init__(self, dim):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()
        )

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GroupNorm(8, dim),
            nn.GELU()
        )

    def forward(self, ir_feat, vis_feat):
        combined = torch.cat([ir_feat, vis_feat], dim=1)
        gate = self.gate_conv(combined)

        # 自适应加权
        fused = gate * ir_feat + (1 - gate) * vis_feat

        # 进一步特征融合
        enhanced = self.feature_fusion(combined)

        return fused + enhanced * 0.5


class MultiBranchEncoder(nn.Module):
    """多分支编码器"""

    def __init__(self, in_ch=1, out_ch=1, ch=64, ch_mult=[1, 2, 4], embed_dim=4,
                 enable_freq_prior=True, enable_aniso_prior=False):
        super().__init__()
        self.depth = len(ch_mult)
        self.enable_freq_prior = enable_freq_prior
        self.enable_aniso_prior = enable_aniso_prior

        # 物理先验模块（复用已有的）
        if self.enable_freq_prior:
            self.freq_encoder_ir = FrequencyEncoder(in_ch, ch, wave='db4')
            self.freq_encoder_vis = FrequencyEncoder(in_ch, ch, wave='db4')

        if self.enable_aniso_prior:
            self.aniso_guidance_ir = AnisotropicGuidanceModule(in_ch, ch // 2)
            self.aniso_guidance_vis = AnisotropicGuidanceModule(in_ch, ch // 2)

        # 共享干
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch, 7, 1, 3),
            nn.GroupNorm(8, ch),
            nn.GELU()
        )

        # 先验融合
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

        # 多分支编码器
        self.ir_branches = nn.ModuleList()
        self.vis_branches = nn.ModuleList()
        self.fusion_modules = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch_mult = [1] + ch_mult
        for i in range(self.depth):
            dim_in = ch * ch_mult[i]
            dim_out = ch * ch_mult[i + 1]

            # 分支模块
            ir_branch = ModalitySpecificBranch(dim_in, dim_in, 'ir')
            vis_branch = ModalitySpecificBranch(dim_in, dim_in, 'vis')
            fusion_module = AdaptiveFusion(dim_in)

            # 下采样
            if i != self.depth - 1:
                downsample = Downsample(dim_in, dim_out)
            else:
                downsample = default_conv(dim_in, dim_out)

            self.ir_branches.append(ir_branch)
            self.vis_branches.append(vis_branch)
            self.fusion_modules.append(fusion_module)
            self.downsamples.append(downsample)

        # 潜在空间卷积
        mid_dim = ch * ch_mult[-1]
        default_act = NonLinearity()
        self.latent_conv1 = ResBlock(dim_in=mid_dim, dim_out=embed_dim, conv=default_conv, act=default_act)
        self.latent_conv2 = ResBlock(dim_in=mid_dim, dim_out=embed_dim, conv=default_conv, act=default_act)

        # 跳跃连接融合（保持与原有架构兼容）
        self.conv_fuse = nn.ModuleList()
        self.num_channels = [ch * ch_mult[0], ch * ch_mult[0], ch * ch_mult[0], ch * ch_mult[1],
                             ch * ch_mult[1], ch * ch_mult[2], ch * ch_mult[2], ch * ch_mult[3], ch * ch_mult[3]]

        for i in range(len(self.num_channels)):
            fuse_layers = [
                nn.Conv2d(self.num_channels[i] * 2, self.num_channels[i], kernel_size=3, padding=1),
                build_norm_layer(dict(type='BN'), self.num_channels[i])[1],
                nn.LeakyReLU(0.2, True),
            ]
            self.conv_fuse.append(nn.Sequential(*fuse_layers))

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x1, x2):
        """
        x1: IR图像
        x2: VIS图像
        """
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

        # 共享干处理
        ir_feat = self.stem(x1)
        vis_feat = self.stem(x2)

        # 先验融合
        if hasattr(self, 'prior_fusion') and (prior_feats_ir or prior_feats_vis):
            target_size = ir_feat.shape[-2:]

            # IR先验融合
            if prior_feats_ir:
                ir_priors_resized = []
                for feat in prior_feats_ir:
                    if feat.shape[-2:] != target_size:
                        feat_resized = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                    else:
                        feat_resized = feat
                    ir_priors_resized.append(feat_resized)

                ir_combined = torch.cat([ir_feat] + ir_priors_resized, dim=1)
                ir_feat = self.prior_fusion(ir_combined)

            # VIS先验融合
            if prior_feats_vis:
                vis_priors_resized = []
                for feat in prior_feats_vis:
                    if feat.shape[-2:] != target_size:
                        feat_resized = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                    else:
                        feat_resized = feat
                    vis_priors_resized.append(feat_resized)

                vis_combined = torch.cat([vis_feat] + vis_priors_resized, dim=1)
                vis_feat = self.prior_fusion(vis_combined)

        # 多分支编码
        h1 = [ir_feat]
        h2 = [vis_feat]
        h = []

        for i in range(self.depth):
            # 分支特化处理
            ir_feat = self.ir_branches[i](ir_feat)
            vis_feat = self.vis_branches[i](vis_feat)

            h1.append(ir_feat)
            h2.append(vis_feat)

            # 自适应融合
            fused_feat = self.fusion_modules[i](ir_feat, vis_feat)

            # 下采样
            ir_feat = self.downsamples[i](ir_feat)
            vis_feat = self.downsamples[i](vis_feat)

        # 潜在空间编码
        x1 = self.latent_conv1(ir_feat)
        x2 = self.latent_conv2(vis_feat)

        # 跳跃连接融合
        for i in range(len(h1)):
            A = torch.cat((h1[i], h2[i]), dim=1)
            x = self.conv_fuse[i](A)
            h.append(x)

        return x1, x2, h, prior_info