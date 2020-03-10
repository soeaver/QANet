import os
import shutil
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from apex import amp
from apex.parallel import DistributedDataParallel

import _init_paths  # pylint: disable=unused-import
from pet.utils.misc import mkdir_p, logging_rank
from pet.utils.net import convert_bn2affine_model, convert_conv2syncbn_model, mismatch_params_filter
from pet.utils.measure import measure_model
from pet.utils.checkpointer import CheckPointer
from pet.utils.optimizer import Optimizer
from pet.utils.lr_scheduler import LearningRateScheduler
from pet.utils.logger import TrainingLogger

from pet.instance.core.config import cfg, merge_cfg_from_file, merge_cfg_from_list
from pet.instance.datasets import build_dataset, make_train_data_loader
from pet.instance.modeling.model_builder import Generalized_CNN

# Parse arguments
parser = argparse.ArgumentParser(description='Pet Model Training')
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional config file',
                    default='./cfgs/instance/mscoco/simple_R-50c-D3K4C256_256x192_adam_1x.yaml', type=str)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('opts', help='See pet/instance/core/config.py for all options',
                    default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()
if args.cfg_file is not None:
    merge_cfg_from_file(args.cfg_file)
if args.opts is not None:
    merge_cfg_from_list(args.opts)

args.device = torch.device(cfg.DEVICE)
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1
if args.distributed:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    args.world_size = torch.distributed.get_world_size()
else:
    args.world_size = 1
    args.local_rank = 0
    cfg.NUM_GPUS = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) if cfg.DEVICE == 'cuda' else 1
    cfg.TRAIN.LOADER_THREADS *= cfg.NUM_GPUS
    cfg.TEST.LOADER_THREADS *= cfg.NUM_GPUS
    cfg.TEST.IMS_PER_GPU *= cfg.NUM_GPUS

logging_rank('Called with args: {}'.format(args), distributed=args.distributed, local_rank=args.local_rank)


def train(model, loader, optimizer, scheduler, logger):
    # switch to train mode
    model.train()
    skip_losses = False if cfg.MODEL.UV_ON else True

    # main loop
    logger.iter_tic()
    logger.data_tic()
    for i, (inputs, targets) in enumerate(loader):
        scheduler.step()  # adjust learning rate
        optimizer.zero_grad()

        inputs = inputs.to(args.device)
        logger.data_toc()

        outputs = model(inputs, targets)
        logger.update_stats(outputs, args.distributed, args.world_size)
        loss = outputs['total_loss']
        if cfg.SOLVER.AMP.ENABLED:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        if args.local_rank == 0:
            logger.log_stats(scheduler.iteration, scheduler.new_lr, skip_losses=False)

        logger.iter_toc()
        logger.iter_tic()
        logger.data_tic()
    return None


def main():
    if not os.path.isdir(cfg.CKPT):
        mkdir_p(cfg.CKPT)
    if args.cfg_file is not None:
        shutil.copyfile(args.cfg_file, os.path.join(cfg.CKPT, args.cfg_file.split('/')[-1]))

    # Create training dataset and loader
    train_set = build_dataset(cfg.TRAIN.DATASETS, is_train=True, local_rank=args.local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if args.distributed else None
    ims_per_gpu = int(cfg.TRAIN.BATCH_SIZE / args.world_size)
    train_loader = make_train_data_loader(train_set, ims_per_gpu, train_sampler)
    cfg.TRAIN.ITER_PER_EPOCH = len(train_loader)
    
    # Calculate FLOPs & Param
    n_flops, n_convops, n_params = -1e6, -1e6, -1e6
    if cfg.CALC_FLOPS and args.local_rank == 0:
        model = Generalized_CNN()
        model.eval()
        n_flops, n_convops, n_params = measure_model(model, cfg.TRAIN.SCALES[0][0], cfg.TRAIN.SCALES[0][1])
        del model

    # Create model
    model = Generalized_CNN()
    logging_rank(model, distributed=args.distributed, local_rank=args.local_rank)
    logging_rank('FLOPs: {:.4f}M, Conv_FLOPs: {:.4f}M, Params: {:.4f}M'.
                 format(n_flops / 1e6, n_convops / 1e6, n_params / 1e6),
                 distributed=args.distributed, local_rank=args.local_rank)
    
    # Create checkpointer
    checkpointer = CheckPointer(cfg.CKPT, weights_path=cfg.TRAIN.WEIGHTS, auto_resume=cfg.TRAIN.AUTO_RESUME,
                                local_rank=args.local_rank)

    # Load model or random-initialization
    model = checkpointer.load_model(model)
    if cfg.MODEL.BATCH_NORM == 'freeze':
        model = convert_bn2affine_model(model, merge=not checkpointer.resume)
    elif cfg.MODEL.BATCH_NORM == 'sync':
        model = convert_conv2syncbn_model(model)
    model.to(args.device)
    if cfg.DEVICE == 'cuda':
        cudnn.benchmark = True
        cudnn.deterministic = False
        cudnn.enabled = True

    # Create optimizer
    optimizer = Optimizer(model, cfg.SOLVER).build()
    optimizer = checkpointer.load_optimizer(optimizer)
    logging_rank('The mismatch keys: {}'.format(mismatch_params_filter(sorted(checkpointer.mismatch_keys))),
                 distributed=args.distributed, local_rank=args.local_rank)
    if cfg.SOLVER.AMP.ENABLED:
        # Create Amp for mixed precision training
        model, optimizer = amp.initialize(model, optimizer, opt_level=cfg.SOLVER.AMP.OPT_LEVEL,
                                          keep_batchnorm_fp32=cfg.SOLVER.AMP.KEEP_BN_FP32,
                                          loss_scale=cfg.SOLVER.AMP.LOSS_SCALE)

    # Create scheduler
    scheduler = LearningRateScheduler(optimizer, cfg.SOLVER, start_iter=0, iter_per_epoch=cfg.TRAIN.ITER_PER_EPOCH,
                                      local_rank=args.local_rank)
    scheduler = checkpointer.load_scheduler(scheduler)

    # Create training logger
    training_logger = TrainingLogger(args.cfg_file.split('/')[-1], scheduler=scheduler, log_period=cfg.DISPLAY_ITER,
                                     iter_per_epoch=cfg.TRAIN.ITER_PER_EPOCH)

    # Model Distributed
    if args.distributed:
        if cfg.SOLVER.AMP.ENABLED:
            model = DistributedDataParallel(model)  # use apex.parallel
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank
            )
    else:
        model = torch.nn.DataParallel(model)

    # Train
    logging_rank('Training starts.', distributed=args.distributed, local_rank=args.local_rank)
    start_epoch = scheduler.iteration // cfg.TRAIN.ITER_PER_EPOCH + 1
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS + 1):
        train_sampler.set_epoch(epoch) if args.distributed else None

        # Train model
        logging_rank('Epoch {} is starting.'.format(epoch), distributed=args.distributed, local_rank=args.local_rank)
        train(model, train_loader, optimizer, scheduler, training_logger)

        # Save model
        if args.local_rank == 0:
            snap_flag = cfg.SOLVER.SNAPSHOT_EPOCHS > 0 and epoch % cfg.SOLVER.SNAPSHOT_EPOCHS == 0
            checkpointer.save(model, optimizer, scheduler, copy_latest=snap_flag, infix='epoch')

    logging_rank('Training done.', distributed=args.distributed, local_rank=args.local_rank)


if __name__ == '__main__':
    main()
