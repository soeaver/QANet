import argparse
import os.path as osp
import sys

from lib.utils.analyser import Analyser
from lib.utils.misc import logging_rank

from instance.core.config import get_cfg, infer_cfg
from instance.modeling.model_builder import Generalized_CNN


sys.path.insert(0, osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), '../../'))


# Parse arguments
parser = argparse.ArgumentParser(description='Pet Model Testing')
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional config file',
                    default='./cfgs/rcnn/mscoco/e2e_faster_rcnn_R-50-FPN_1x.yaml', type=str)
parser.add_argument("--size", type=int, nargs=2)
parser.add_argument('opts', help='See pet/rcnn/core/config.py for all options',
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()


def main():
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg = infer_cfg(cfg)
    cfg.freeze()

    # Calculate Params & FLOPs & Activations
    if cfg.MODEL_ANALYSE:
        model = Generalized_CNN(cfg)
        model.eval()

        analyser = Analyser(cfg, model, param_details=False)

        n_params = analyser.get_params()[1]
        conv_flops, model_flops = analyser.get_flops_activs(args.size[0], args.size[1], mode='flops')
        conv_activs, model_activs = analyser.get_flops_activs(args.size[0], args.size[1], mode='activations')

        logging_rank('-----------------------------------')
        logging_rank('Params: {}'.format(n_params))
        logging_rank('FLOPs: {:.4f} M / Conv_FLOPs: {:.4f} M'.format(model_flops, conv_flops))
        logging_rank('ACTIVATIONs: {:.4f} M / Conv_ACTIVATIONs: {:.4f} M'.format(model_activs, conv_activs))
        logging_rank('-----------------------------------')

        del model


if __name__ == '__main__':
    main()
