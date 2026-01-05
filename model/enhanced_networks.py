import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

logger = logging.getLogger('base')

# 导入原有的初始化函数（保持不变）
from model.networks import (
    weights_init_normal, weights_init_kaiming, weights_init_orthogonal, init_weights
)


def define_Enhanced_Network(opt, local_rank):
    """定义增强版网络（集成物理先验）"""
    from model.head.fusion_pipiline import Fusion_Pipiline

    # 网络配置
    model_opt = opt['model']
    Unet_opt = model_opt['Unet']

    # 增强版编码器/解码器
    if Unet_opt['LDM'] == 'Enhanced_U_LDM':
        from model.diffusers.enhanced_UNet_arch import EnhancedEncode, EnhancedDecode

        # 物理先验配置
        enable_freq_prior = Unet_opt.get('enable_freq_prior', True)
        enable_aniso_prior = Unet_opt.get('enable_aniso_prior', True)

        LDM_Enc = EnhancedEncode(
            in_ch=Unet_opt['in_ch'],
            out_ch=Unet_opt['out_ch'],
            ch=Unet_opt['LDM_ch'],
            ch_mult=Unet_opt['ch_mult'],
            embed_dim=Unet_opt['LDM_embed_dim'],
            enable_freq_prior=enable_freq_prior,
            enable_aniso_prior=enable_aniso_prior
        )

        LDM_Dec = EnhancedDecode(
            in_ch=Unet_opt['in_ch'],
            out_ch=Unet_opt['out_ch'],
            ch=Unet_opt['LDM_ch'],
            ch_mult=Unet_opt['ch_mult'],
            embed_dim=Unet_opt['LDM_embed_dim']
        )
    else:
        # 回退到原有架构
        from model.diffusers.UNet_arch import Encode, Decode
        LDM_Enc = Encode(
            in_ch=Unet_opt['in_ch'],
            out_ch=Unet_opt['out_ch'],
            ch=Unet_opt['LDM_ch'],
            ch_mult=Unet_opt['ch_mult'],
            embed_dim=Unet_opt['LDM_embed_dim']
        )
        LDM_Dec = Decode(
            in_ch=Unet_opt['in_ch'],
            out_ch=Unet_opt['out_ch'],
            ch=Unet_opt['LDM_ch'],
            ch_mult=Unet_opt['ch_mult'],
            embed_dim=Unet_opt['LDM_embed_dim']
        )

    # 扩散融合模型（保持不变）
    Fusion_model = Fusion_Pipiline(
        LDM_Enc,
        LDM_Dec,
        mode=model_opt['Fusion']['mode'],
        fusion_task=opt['model']['fusion_task'],
        sample_selected=model_opt['Fusion']['sample_selected'],
        model_selected=model_opt['Fusion']['model_selected'],
        feat_channels=model_opt['Fusion']['feat_channels'],
        inference_steps=model_opt['Fusion']['inference_steps'],
        num_train_timesteps=model_opt['Fusion']['num_train_timesteps'],
        channel_emdin=model_opt['Fusion']['channel_emdin'],
        num_blocks=model_opt['Fusion']['num_blocks'],
        heads=model_opt['Fusion']['heads']
    )

    # 初始化权重
    if opt['phase'] == 'train':
        load_path = opt['path']['resume_state']
        if load_path is None:
            init_weights(Fusion_model.model, init_type='normal')
            init_weights(Fusion_model.fusion, init_type='normal')

    # 分布式训练支持
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        Fusion_model = nn.SyncBatchNorm.convert_sync_batchnorm(Fusion_model)
        Fusion_model = nn.parallel.DistributedDataParallel(
            Fusion_model.cuda(local_rank),
            device_ids=[local_rank],
            find_unused_parameters=True
        )

    return Fusion_model