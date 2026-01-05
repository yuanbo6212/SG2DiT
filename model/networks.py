import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

logger = logging.getLogger('base')


##########################################
#                    Weight initialization
##########################################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


##########################################
#                define Network
##########################################


# Generator
def define_Network(opt, local_rank):
    from model.head.fusion_pipiline import Fusion_Pipiline

    # latent space
    model_opt = opt['model']
    Unet_opt = model_opt['Unet']

    # === 核心修改：根据配置选择UNet架构 ===
    if Unet_opt['LDM'] == 'U_LDM':
        # 原有基础架构
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

    elif Unet_opt['LDM'] == 'Enhanced_U_LDM':
        # 增强架构（带物理先验）
        from model.diffusers.enhanced_UNet_arch import EnhancedEncode, EnhancedDecode

        # 物理先验配置（从配置文件读取，有默认值）
        enable_freq_prior = Unet_opt.get('enable_freq_prior', True)
        enable_aniso_prior = Unet_opt.get('enable_aniso_prior', False)

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

    elif Unet_opt['LDM'] == 'SwinConv_LDM':
        # Swin-Conv混合架构
        from model.diffusers.swin_conv_encoder import SwinConvEncoder
        from model.diffusers.enhanced_UNet_arch import EnhancedDecode  # 复用增强解码器

        # 物理先验配置
        enable_freq_prior = Unet_opt.get('enable_freq_prior', True)
        enable_aniso_prior = Unet_opt.get('enable_aniso_prior', False)

        # Swin-Conv特有参数
        window_size = Unet_opt.get('window_size', 7)
        num_heads = Unet_opt.get('num_heads', 8)

        LDM_Enc = SwinConvEncoder(
            in_ch=Unet_opt['in_ch'],
            out_ch=Unet_opt['out_ch'],
            ch=Unet_opt['LDM_ch'],
            ch_mult=Unet_opt['ch_mult'],
            embed_dim=Unet_opt['LDM_embed_dim'],
            enable_freq_prior=enable_freq_prior,
            enable_aniso_prior=enable_aniso_prior,
            window_size=window_size,
            num_heads=num_heads
        )

        LDM_Dec = EnhancedDecode(
            in_ch=Unet_opt['in_ch'],
            out_ch=Unet_opt['out_ch'],
            ch=Unet_opt['LDM_ch'],
            ch_mult=Unet_opt['ch_mult'],
            embed_dim=Unet_opt['LDM_embed_dim']
        )

    elif Unet_opt['LDM'] == 'MultiBranch_LDM':
        # 多分支架构
        try:
            from model.diffusers.multi_branch_encoder import MultiBranchEncoder
            from model.diffusers.enhanced_UNet_arch import EnhancedDecode

            # 物理先验配置
            enable_freq_prior = Unet_opt.get('enable_freq_prior', True)
            enable_aniso_prior = Unet_opt.get('enable_aniso_prior', False)

            LDM_Enc = MultiBranchEncoder(
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
        except ImportError:
            logger.warning("MultiBranchEncoder not implemented, falling back to Enhanced_U_LDM")
            from model.diffusers.enhanced_UNet_arch import EnhancedEncode, EnhancedDecode

            enable_freq_prior = Unet_opt.get('enable_freq_prior', True)
            enable_aniso_prior = Unet_opt.get('enable_aniso_prior', False)

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

    elif Unet_opt['LDM'] == 'DT_LDM':
        # Diffusion Transformer架构
        try:
            from model.diffusers.diffusion_transformer_encoder import DTEncoder
            from model.diffusers.enhanced_UNet_arch import EnhancedDecode

            # 物理先验配置
            enable_freq_prior = Unet_opt.get('enable_freq_prior', True)
            enable_aniso_prior = Unet_opt.get('enable_aniso_prior', False)

            # Transformer特有参数
            patch_size = Unet_opt.get('patch_size', 8)
            num_layers = Unet_opt.get('num_layers', 6)
            num_heads = Unet_opt.get('num_heads', 8)

            LDM_Enc = DTEncoder(
                in_ch=Unet_opt['in_ch'],
                out_ch=Unet_opt['out_ch'],
                ch=Unet_opt['LDM_ch'],
                ch_mult=Unet_opt['ch_mult'],
                embed_dim=Unet_opt['LDM_embed_dim'],
                enable_freq_prior=enable_freq_prior,
                enable_aniso_prior=enable_aniso_prior,
                patch_size=patch_size,
                num_layers=num_layers,
                num_heads=num_heads
            )

            LDM_Dec = EnhancedDecode(
                in_ch=Unet_opt['in_ch'],
                out_ch=Unet_opt['out_ch'],
                ch=Unet_opt['LDM_ch'],
                ch_mult=Unet_opt['ch_mult'],
                embed_dim=Unet_opt['LDM_embed_dim']
            )
        except ImportError:
            logger.warning("DTEncoder not implemented, falling back to Enhanced_U_LDM")
            from model.diffusers.enhanced_UNet_arch import EnhancedEncode, EnhancedDecode

            enable_freq_prior = Unet_opt.get('enable_freq_prior', True)
            enable_aniso_prior = Unet_opt.get('enable_aniso_prior', False)

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
        raise ValueError(
            f"Unknown LDM type: {Unet_opt['LDM']}. Supported types: U_LDM, Enhanced_U_LDM, SwinConv_LDM, MultiBranch_LDM, DT_LDM")

    # 打印编码器信息
    logger.info(f"Using LDM encoder: {Unet_opt['LDM']}")
    if hasattr(LDM_Enc, 'enable_freq_prior'):
        logger.info(f"Frequency prior: {LDM_Enc.enable_freq_prior}")
    if hasattr(LDM_Enc, 'enable_aniso_prior'):
        logger.info(f"Anisotropic prior: {LDM_Enc.enable_aniso_prior}")

    # Diffusion fusion model
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

    if opt['phase'] == 'train':
        load_path = opt['path']['resume_state']

        if load_path is None:
            init_weights(Fusion_model.model, init_type='normal')
            init_weights(Fusion_model.fusion, init_type='normal')

    # 打印模型参数信息
    total_params = sum(p.numel() for p in Fusion_model.parameters())
    trainable_params = sum(p.numel() for p in Fusion_model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Determine whether to enable distributed multi-GPU computing
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        # BN layer synchronization
        Fusion_model = nn.SyncBatchNorm.convert_sync_batchnorm(Fusion_model)
        # Put a copy of the model into the DDP
        Fusion_model = nn.parallel.DistributedDataParallel(
            Fusion_model.cuda(local_rank),
            device_ids=[local_rank],
            find_unused_parameters=True
        )

    return Fusion_model