import os
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import time
import random
import numpy as np
import json
from math import ceil
from util.visualizer import Visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/train.json', help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--sample_selected', type=str, default=None, help='select diffusion sampler')
    parser.add_argument('--model_selected', type=str, default=None, help='select diffusion network')
    parser.add_argument('--batch_size', type=int, default=None, help='set batch_size')
    parser.add_argument('--fusion_task', type=str, default=None, help='set fusion_task')
    parser.add_argument('--strategy', type=str, default=None, help='set fusion strategy')

    # =====================================================================
    #           First：Set parameters
    # ======================================================================

    args = parser.parse_args()
    args.gpu_ids = str(args.gpu)


    # --- begin debug & robust parse patch ---
    # parse config via logger.parse
    opt = Logger.parse(args)

    # print which config path was used (help debug)
    print(">>> Logger.parse used config path: ", args.config)

    # show the top-level model content returned by parse (for debug)
    print(">>> opt['model'] from Logger.parse (raw):")
    try:
        print(json.dumps(opt.get('model', {}), indent=2))
    except Exception:
        print(str(opt.get('model', {})))

    # If Fusion is missing/None, try to re-load the JSON directly from file (defensive fix)
    fusion_ok = False
    try:
        if opt.get('model') and opt['model'].get('Fusion') and opt['model']['Fusion'].get('mode') is not None:
            fusion_ok = True
    except Exception:
        fusion_ok = False

    if not fusion_ok:
        # Attempt to load config file directly to recover model.Fusion
        cfg_path = args.config if args.config is not None else None
        if cfg_path is None:
            raise RuntimeError(
                "Config path is None and opt['model']['Fusion'] is missing. Please provide --config <path>.")
        print(f">>> opt['model']['Fusion'] is missing or None — attempting to reload JSON from {cfg_path}")
        try:
            with open(cfg_path, 'r') as f:
                # remove inline comments starting with // (same logic as logger.parse)
                json_str = ''
                for line in f:
                    json_str += line.split('//')[0] + '\n'
                cfg = json.loads(json_str)
            # if cfg contains model.Fusion, copy it into opt
            if 'model' in cfg and 'Fusion' in cfg['model']:
                opt.setdefault('model', {})['Fusion'] = cfg['model']['Fusion']
                opt['model']['fusion_task'] = cfg['model'].get('fusion_task', opt['model'].get('fusion_task', None))
                print(">>> Reload successful — updated opt['model']['Fusion'] from file.")
                print(">>> New opt['model']['Fusion']:")
                print(json.dumps(opt['model']['Fusion'], indent=2))
                fusion_ok = True
            else:
                print(">>> Reloaded JSON does not contain model.Fusion. Please check the file content.")
        except Exception as e:
            print(">>> Failed to reload config JSON directly:", repr(e))
            raise

    # finally convert to NoneDict (as before)
    opt = Logger.dict_to_nonedict(opt)
    # --- patch: fix Fusion None override issue ---
    if opt.get('model') and opt['model'].get('Fusion'):
        f = opt['model']['Fusion']
        # If mode/model_selected/sample_selected become None, restore them
        if f.get('mode') is None or f.get('model_selected') is None:
            print(">>> [Patch] Fusion parameters were lost after dict_to_nonedict, restoring manually.")
            f['mode'] = f.get('mode') or 'MAX'
            f['model_selected'] = f.get('model_selected') or 'DFT'
            f['sample_selected'] = f.get('sample_selected') or 'ddp-solver++'
            opt['model']['Fusion'] = f
            print(">>> [Patch] Restored Fusion config:", f)
    # --- end patch ---

    # --- end debug & robust parse patch ---

    # ---------------- safe override: only override opt with args when args is not None ----------------
    # Fusion sample/model selection
    if getattr(args, 'sample_selected', None) is not None:
        opt['model']['Fusion']['sample_selected'] = args.sample_selected
    if getattr(args, 'model_selected', None) is not None:
        opt['model']['Fusion']['model_selected'] = args.model_selected
    # batch size override for dataset
    if getattr(args, 'batch_size', None) is not None:
        opt['datasets']['train']['batch_size'] = args.batch_size
    # fusion task
    if getattr(args, 'fusion_task', None) is not None:
        opt['model']['fusion_task'] = args.fusion_task
    # fusion strategy / mode
    if getattr(args, 'strategy', None) is not None:
        opt['model']['Fusion']['mode'] = args.strategy
    # -----------------------------------------------------------------------------------------------

    # ---------------- GPU setting ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    print(f"Using single GPU: {device}")

    # ---------------- Logger setup ----------------
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # =====================================================================
    #                    Read dataset
    # =====================================================================
    phase = 'train'
    dataset_opt = opt['datasets']['train']
    batchSize = dataset_opt['batch_size']

    train_set = Data.create_dataset_Fusion(dataset_opt, phase, opt)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batchSize,
        shuffle=dataset_opt['use_shuffle'],
        num_workers=dataset_opt['num_workers'],
        pin_memory=True,
    )

    training_iters = int(ceil(train_set.length / float(batchSize)))
    original_list = opt['train']['scheduler']['periods']
    result_list = [x * training_iters for x in original_list]
    opt['train']['scheduler']['periods'] = result_list

    logger.info('Initial Dataset Finished')

    # ---------------- Random seed ----------------
    seed = random.randint(0, 10000)
    print("Current random seed: ", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # =====================================================================
    #                    Create model
    # =====================================================================
    diffusion = Model.create_model(opt, device)
    logger.info('Initial Model Finished')


    if hasattr(diffusion, "Fusion_net"):
        diffusion.Fusion_net = diffusion.Fusion_net.to(device)
    if hasattr(diffusion, "Discriminator"):
        diffusion.Discriminator = diffusion.Discriminator.to(device)
    logger.info('✅ Model moved to GPU successfully.')



    visualizer = Visualizer(opt)
    current_step = diffusion.begin_step
    start_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']

    if opt['path']['resume_state']:
        print(f'Resuming training from epoch: {start_epoch}, iter: {current_step}.')

    # =====================================================================
    #                           Train
    # =====================================================================
    for current_epoch in range(start_epoch, n_epoch):
        for istep, train_data in enumerate(train_loader):
            iter_start_time = time.time()
            current_step += 1

            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            diffusion.update_learning_rate(current_epoch * train_set.length + istep * batchSize,
                                           warmup_iter=opt['train'].get('warmup_iter', -1))

            # Logging
            if (istep + 1) % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                t = (time.time() - iter_start_time) / batchSize
                lr_log = diffusion.get_current_learning_rate()
                visualizer.print_current_errors(current_epoch, istep + 1, training_iters, logs, lr_log[0], 'Train')
                visuals = diffusion.get_current_visuals()
                visualizer.display_current_results(visuals, current_epoch, True)

            # Validation
            if (istep + 1) % opt['train']['val_freq'] == 0:
                result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                os.makedirs(result_path, exist_ok=True)
                diffusion.test_fusion()
                visuals = diffusion.get_current_test()
                visualizer.display_current_results(visuals, current_epoch, True)

        if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
            diffusion.save_network(current_epoch, current_step)

    print("Training completed successfully.")
