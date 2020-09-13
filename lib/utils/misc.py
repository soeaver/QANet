import errno
import os
import sys
import logging
import functools
import numpy as np
from six.moves import cPickle as pickle

import torch
import torch.distributed as dist

from .comm import is_main_process

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("misc")
logger.propagate = False
plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m-%d %H:%M:%S"
    )


# Set up logging and load config options
def setup_logging(path, local_logger=None, local_plain_formatter=None):
    """Collect logger information"""
    if not os.path.isdir(path):
        mkdir_p(path)
    filename = os.path.join(path, "log.txt")
    fh = logging.StreamHandler(cached_log_stream(filename))
    fh.setLevel(logging.DEBUG)
    formatter = plain_formatter if local_plain_formatter is None else local_plain_formatter
    fh.setFormatter(formatter)
    ch = None
    if is_main_process():
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
    if local_logger is None:
        logger.addHandler(fh)
        if ch is not None:
            logger.addHandler(ch)
        return logger
    else:
        local_logger.addHandler(fh)
        if ch is not None:
            local_logger.addHandler(ch)
        return local_logger


@functools.lru_cache(maxsize=None)
def cached_log_stream(filename):
    if not os.path.exists(filename):
        return open(filename, "w")
    return open(filename, "a")


def logging_rank(sstr, local_logger=None):
    if is_main_process():
        if local_logger is not None:
            local_logger.info(sstr)
        else:
            logger.info(sstr)
    return 0


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def mkdir_p(path):
    """make dir if not exist"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_object(obj, file_name):
    """Save a Python object by pickling it."""
    file_name = os.path.abspath(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
            box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


def concat_rotatedbox_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
            box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax5 = box_regression_per_level.shape[1]
        A = Ax5 // 5
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 5, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 5)
    return box_cls, box_regression
