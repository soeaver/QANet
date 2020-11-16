import argparse
import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp

import _init_paths  # pylint: disable=unused-import
from lib.utils.analyser import Analyser
from lib.utils.checkpointer import CheckPointer
from lib.utils.comm import all_gather, get_world_size, is_main_process, synchronize
from lib.utils.events import EventStorage
from lib.utils.logger import build_test_hooks, build_train_hooks, write_metrics
from lib.utils.lr_scheduler import LearningRateScheduler
from lib.utils.misc import logging_rank, mkdir_p, setup_logging
from lib.utils.net import mismatch_params_filter
from lib.utils.optimizer import Optimizer
from lib.utils.timer import Timer

from qanet.core.config import get_cfg, infer_cfg
from qanet.core.test import TestEngine
from qanet.datasets.dataset import build_dataset, make_data_sampler, make_test_data_loader, make_train_data_loader
from qanet.datasets.evaluation import Evaluation
from qanet.modeling.model_builder import Generalized_CNN


def train(cfg, model, sampler, train_loader, test_loader, test_set, test_engine,
          optimizer, scheduler, scaler, checkpointer, all_hooks):
    # switch to train mode
    model.train()
    iter_per_epoch = len(train_loader)
    start_iter = scheduler.iteration
    start_epoch = start_iter // iter_per_epoch + 1
    # main loop
    with EventStorage(start_iter=start_iter, log_period=cfg.DISPLAY_ITER) as storage:
        try:
            for h in all_hooks:
                h.before_train()

            for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS + 1):
                logging_rank("Starting training from epoch {}".format(epoch))
                sampler.set_epoch(epoch)
                iter_loader = iter(enumerate(train_loader, start_iter))
                for iteration in range(0, iter_per_epoch):
                    for h in all_hooks:
                        h.before_step(storage=storage)

                    data_start = time.perf_counter()
                    _, (inputs, targets) = next(iter_loader)
                    inputs = inputs.to(torch.device(cfg.DEVICE))
                    data_time = time.perf_counter() - data_start

                    optimizer.zero_grad()

                    with amp.autocast(enabled=cfg.SOLVER.AMP.ENABLED):
                        outputs = model(inputs, targets)
                        losses = sum(outputs['losses'].values())

                    metrics_dict = outputs['losses']
                    metrics_dict["data_time"] = data_time
                    metrics_dict["best_acc1"] = scheduler.info['best_acc']
                    write_metrics(metrics_dict, storage)

                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    for h in all_hooks:
                        h.after_step(epoch=epoch, storage=storage)
                    storage.step()

                # Save model
                if is_main_process():
                    snap_flag = cfg.SOLVER.SNAPSHOT_EPOCHS > 0 and epoch % cfg.SOLVER.SNAPSHOT_EPOCHS == 0
                    checkpointer.save(model, optimizer, scheduler, copy_latest=snap_flag, infix='epoch')

                # Test model
                if cfg.TEST.DATASETS and cfg.TEST.FREQUENCY_EPOCHS != -1 and epoch % cfg.TEST.FREQUENCY_EPOCHS == 0:
                    if not os.path.isdir(os.path.join(cfg.CKPT, 'test')):
                        mkdir_p(os.path.join(cfg.CKPT, 'test'))
                    scheduler.info['cur_epoch'] = epoch
                    cur_acc = test(cfg, test_engine, test_loader, test_set)
                    if is_main_process():
                        scheduler.info['cur_acc'] = cur_acc * 100
                        logging_rank(
                            'Evaluating at epoch {} is done, acc1 is {:4.2f}%.'.format(epoch, scheduler.info['cur_acc'])
                        )
                        checkpointer.save_best(model, optimizer, scheduler)
                    synchronize()
                    model.train()
        finally:
            for h in all_hooks:
                h.after_train(storage=storage)
    return None


def test(cfg, test_engine, loader, datasets):
    # switch to evaluate mode
    test_engine.model.eval()

    all_hooks = build_test_hooks(args.cfg_file.split('/')[-1], log_period=10)

    total_timer = Timer()
    total_timer.tic()
    all_results = [[] for _ in range(4)]
    eval = Evaluation(cfg, training=True)
    with torch.no_grad():
        loader = iter(loader)
        for i in range(len(loader)):
            all_hooks.iter_tic()

            all_hooks.data_tic()
            inputs, targets, idx = next(loader)
            all_hooks.data_toc()

            all_hooks.infer_tic()
            result = test_engine(inputs, targets)
            all_hooks.infer_toc()

            all_hooks.post_tic()
            eval_results = eval.post_processing(result, targets, idx, datasets)
            all_results = [results + eva for results, eva in zip(all_results, eval_results)]
            all_hooks.post_toc()

            all_hooks.iter_toc()
            if is_main_process():
                all_hooks.log_stats(i, 0, len(loader), len(datasets))

    all_results = list(zip(*all_gather(all_results)))
    all_results = [[item for sublist in results for item in sublist] for results in all_results]
    if is_main_process():
        total_timer.toc(average=False)
        logging_rank('Total inference time: {:.3f}s'.format(total_timer.average_time))
        return eval.evaluation(datasets, all_results)
    else:
        return 0.


def main(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg = infer_cfg(cfg)
    cfg.freeze()
    # logging_rank(cfg)

    if not os.path.isdir(cfg.CKPT):
        mkdir_p(cfg.CKPT)
    setup_logging(cfg.CKPT)
    if args.cfg_file is not None:
        shutil.copyfile(args.cfg_file, os.path.join(cfg.CKPT, args.cfg_file.split('/')[-1]))

    # Calculate Params & FLOPs & Activations
    n_params, conv_flops, model_flops, conv_activs, model_activs = 0, 0, 0, 0, 0
    if is_main_process() and cfg.MODEL_ANALYSE:
        model = Generalized_CNN(cfg)
        model.eval()
        analyser = Analyser(cfg, model, param_details=False)
        n_params = analyser.get_params()[1]
        conv_flops, model_flops = analyser.get_flops_activs(cfg.TRAIN.SCALES[0][0], cfg.TRAIN.SCALES[0][1],
                                                            mode='flops')
        conv_activs, model_activs = analyser.get_flops_activs(cfg.TRAIN.SCALES[0][0], cfg.TRAIN.SCALES[0][1],
                                                              mode='activations')
        del model

    synchronize()
    # Create model
    model = Generalized_CNN(cfg)
    logging_rank(model)
    logging_rank(
        'Params: {} | FLOPs: {:.4f}M / Conv_FLOPs: {:.4f}M | '
        'ACTIVATIONs: {:.4f}M / Conv_ACTIVATIONs: {:.4f}M'.format(
            n_params, model_flops, conv_flops, model_activs, conv_activs
        )
    )

    # Create checkpointer
    checkpointer = CheckPointer(cfg.CKPT, weights_path=cfg.TRAIN.WEIGHTS, auto_resume=cfg.TRAIN.AUTO_RESUME)

    # Load model or random-initialization
    model = checkpointer.load_model(model)
    model.to(torch.device(cfg.DEVICE))
    if cfg.DEVICE == 'cuda' and cfg.CUDNN:
        cudnn.benchmark = True
        cudnn.enabled = True

    # Create optimizer
    optimizer = Optimizer(model, cfg.SOLVER).build()
    optimizer = checkpointer.load_optimizer(optimizer)
    logging_rank('The mismatch keys: {}'.format(mismatch_params_filter(sorted(checkpointer.mismatch_keys))))

    # Create training loader
    datasets = build_dataset(cfg, is_train=True)
    train_sampler = make_data_sampler(cfg, datasets)
    train_loader = make_train_data_loader(cfg, datasets, train_sampler)
    iter_per_epoch = len(train_loader)

    # Create testing dataset, loader and engine
    if cfg.TEST.DATASETS:
        test_set = build_dataset(cfg, is_train=False)
        test_loader = make_test_data_loader(cfg, test_set)
        test_engine = TestEngine(cfg, model)
    else:
        test_set, test_loader, test_engine = None, None, None
    synchronize()

    # Create scheduler
    scheduler = LearningRateScheduler(optimizer, cfg.SOLVER, start_iter=0, iter_per_epoch=iter_per_epoch)
    scheduler = checkpointer.load_scheduler(scheduler)

    # Model Distributed
    distributed = get_world_size() > 1
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    # Create Amp for mixed precision training
    scaler = amp.GradScaler(enabled=cfg.SOLVER.AMP.ENABLED)

    # Build hooks
    max_iter = iter_per_epoch
    warmup_iter = iter_per_epoch * cfg.SOLVER.WARM_UP_EPOCH
    all_hooks = build_train_hooks(cfg, optimizer, scheduler, max_iter=max_iter, warmup_iter=warmup_iter,
                                  ignore_warmup_time=False)

    # Train
    train(cfg, model, train_sampler, train_loader, test_loader, test_set, test_engine,
          optimizer, scheduler, scaler, checkpointer, all_hooks)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='QANet Model Training')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./cfgs/CIHP/QANet/QANet_R-50c_512x384_1x.yaml', type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('opts', help='See qanet/core/config.py for all options',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )

    main(args)
