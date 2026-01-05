'''create dataset and dataloader'''
import logging


#================================================================
#                         load dataset
#=================================================================
def create_dataset_Fusion(dataset_opt, phase, opt):
    logger = logging.getLogger('base')

    # ===============================
    # 1. 检查 fusion_task 是否定义
    # ===============================
    fusion_task = opt['model'].get('fusion_task', None)
    if fusion_task is None:
        logger.warning("⚠️ 'fusion_task' not specified in model config, using default VI-IR task.")
        fusion_task = 'VI-IR'

    # ===============================
    # 2. 根据任务类型确定路径
    # ===============================
    path1, path2 = None, None
    try:
        if fusion_task == 'VI-IR':
            path1 = dataset_opt.get('dataroot_ir')
            path2 = dataset_opt.get('dataroot_vi')
        elif fusion_task == 'VI-NIR':
            path1 = dataset_opt.get('dataroot_NIR')
            path2 = dataset_opt.get('dataroot_Vis')
        elif fusion_task == 'PIF':
            path1 = dataset_opt.get('dataroot_DoLP')
            path2 = dataset_opt.get('dataroot_S0')
        elif fusion_task == 'MEF':
            path1 = dataset_opt.get('dataroot_over')
            path2 = dataset_opt.get('dataroot_under')
        elif fusion_task == 'MFF':
            path1 = dataset_opt.get('dataroot_source_1')
            path2 = dataset_opt.get('dataroot_source_2')
        elif fusion_task == 'SPECT-MRI':
            path1 = dataset_opt.get('dataroot_SPECT_MRI')
            path2 = dataset_opt.get('dataroot_SPECT')
        elif fusion_task == 'PET-MRI':
            path1 = dataset_opt.get('dataroot_PET_MRI')
            path2 = dataset_opt.get('dataroot_PET')
        else:
            raise ValueError(f"Unknown fusion_task: {fusion_task}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to parse dataset paths for fusion_task={fusion_task}: {e}")

    # ===============================
    # 3. 检查 Fusion 模式
    # ===============================
    fusion_mode = opt['model'].get('Fusion', {}).get('mode', 'MAX')
    dataset = None

    if fusion_mode == 'MAX':
        try:
            from data.Fusion_dataset import FusionDataset
            dataset = FusionDataset(
                split=phase,
                crop_size=dataset_opt.get('crop_size', 256),
                min_max=(-1, 1),
                ir_path=path1,
                vi_path=path2,
                is_crop=dataset_opt.get('is_crop', False),
            )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to create FusionDataset (MAX mode): {e}")

    elif fusion_mode == 'MEAN':
        try:
            from data.Fusion_dataset import FusionDataset_Digtal
            dataset = FusionDataset_Digtal(
                split=phase,
                crop_size=dataset_opt.get('crop_size', 256),
                min_max=(-1, 1),
                img1_path=path1,
                img2_path=path2,
                is_crop=dataset_opt.get('is_crop', False),
            )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to create FusionDataset_Digtal (MEAN mode): {e}")
    else:
        raise NotImplementedError(f"❌ Unknown Fusion mode: {fusion_mode}")

    # ===============================
    # 4. 日志输出与返回
    # ===============================
    if dataset is None:
        raise RuntimeError("❌ Dataset creation failed — no valid dataset instance created.")

    logger.info(f"✅ Dataset [{dataset.__class__.__name__} - {dataset_opt['name']}] is created successfully.")
    return dataset
