import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DTEncoder(nn.Module):
    """简化的Diffusion Transformer编码器"""

    def __init__(self, in_ch=1, out_ch=1, ch=64, ch_mult=[1, 2, 4], embed_dim=4,
                 enable_freq_prior=True, enable_aniso_prior=False):
        super().__init__()
        # 暂时继承原有UNet结构，后续扩展
        from model.diffusers.enhanced_UNet_arch import EnhancedEncode
        self.base_encoder = EnhancedEncode(
            in_ch=in_ch, out_ch=out_ch, ch=ch, ch_mult=ch_mult, embed_dim=embed_dim,
            enable_freq_prior=enable_freq_prior, enable_aniso_prior=enable_aniso_prior
        )

    def forward(self, x1, x2):
        return self.base_encoder(x1, x2)