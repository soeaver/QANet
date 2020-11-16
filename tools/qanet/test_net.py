import argparse
import os

import torch

import _init_paths  # pylint: disable=unused-import
from lib.utils.analyser import Analyser
from lib.utils.checkpointer import get_weights, load_weights
from lib.utils.comm import all_gather, is_main_process, synchronize
from lib.utils.logger import build_test_hooks
from lib.utils.misc import logging_rank, mkdir_p, setup_logging
from lib.utils.timer import Timer

from qanet.core.config import get_cfg, infer_cfg
from qanet.core.test import TestEngine
from qanet.datasets.dataset import build_dataset, make_test_data_loader
from qanet.datasets.evaluation import Evaluation
from qanet.modeling.model_builder import Generalized_CNN


def test(cfg, test_engine, loader, datasets, all_hooks):
    total_timer = Timer()
    total_timer.tic()
    all_results = [[] for _ in range(4)]
    eval = Evaluation(cfg)
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
        eval.evaluation(datasets, all_results)


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

    # Calculate Params & FLOPs & Activations
    n_params, conv_flops, model_flops, conv_activs, model_activs = 0, 0, 0, 0, 0
    if is_main_process() and cfg.MODEL_ANALYSE:
        model = Generalized_CNN(cfg)
        model.eval()
        analyser = Analyser(cfg, model, param_details=False)
        n_params = analyser.get_params()[1]
        conv_flops, model_flops = analyser.get_flops_activs(cfg.TEST.SCALE[0], cfg.TEST.SCALE[1], mode='flops')
        conv_activs, model_activs = analyser.get_flops_activs(cfg.TEST.SCALE[0], cfg.TEST.SCALE[1], mode='activations')
        del model

    synchronize()
    # Create model
    model = Generalized_CNN(cfg)
    logging_rank(model)

    # Load model
    test_weights = get_weights(cfg.CKPT, cfg.TEST.WEIGHTS)
    load_weights(model, test_weights)
    logging_rank(
        'Params: {} | FLOPs: {:.4f}M / Conv_FLOPs: {:.4f}M | '
        'ACTIVATIONs: {:.4f}M / Conv_ACTIVATIONs: {:.4f}M'.format(
            n_params, model_flops, conv_flops, model_activs, conv_activs
        )
    )

    model.eval()
    model.to(torch.device(cfg.DEVICE))

    # Create testing dataset and loader
    datasets = build_dataset(cfg, is_train=False)
    test_loader = make_test_data_loader(cfg, datasets)
    synchronize()

    # Build hooks
    all_hooks = build_test_hooks(args.cfg_file.split('/')[-1], log_period=1, num_warmup=0)

    # Build test engine
    test_engine = TestEngine(cfg, model)

    # Test
    test(cfg, test_engine, test_loader, datasets, all_hooks)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='QANet Model Training')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./ckpts/CIHP/QANet/QANet_R-50c_512x384_1x/QANet_R-50c_512x384_1x.yaml', type=str)
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
