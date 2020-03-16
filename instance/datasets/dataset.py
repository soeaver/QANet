import os

import torch.utils.data

from utils.data import datasets as D
from utils.data import samplers
from utils.misc import logging_rank
from instance.datasets import build_transforms
from instance.datasets.dataset_catalog import contains, get_im_dir, get_ann_fn
from instance.core.config import cfg


def build_dataset(dataset_list, is_train=True, local_rank=0):
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    for dataset_name in dataset_list:
        assert contains(dataset_name), 'Unknown dataset name: {}'.format(dataset_name)
        assert os.path.exists(get_im_dir(dataset_name)), 'Im dir \'{}\' not found'.format(get_im_dir(dataset_name))
        logging_rank('Creating: {}'.format(dataset_name), local_rank=local_rank)

    transforms = build_transforms(is_train)
    datasets = []
    for dataset_name in dataset_list:
        args = {}
        args['root'] = get_im_dir(dataset_name)
        args['ann_file'] = get_ann_fn(dataset_name)
        args['bbox_file'] = cfg.TEST.INSTANCE_BBOX_FILE
        args['image_thresh'] = cfg.TEST.IMAGE_THRESH
        ann_types = ()
        if cfg.MODEL.MASK_ON:
            ann_types = ann_types + ('mask',)
        if cfg.MODEL.KEYPOINT_ON:
            ann_types = ann_types + ('keypoints',)
        if cfg.MODEL.PARSING_ON or cfg.MODEL.QANET_ON:
            ann_types = ann_types + ('parsing',)
        if cfg.MODEL.UV_ON:
            ann_types = ann_types + ('uv',)
        args['ann_types'] = ann_types
        args["transforms"] = transforms
        dataset = D.COCOInstanceDataset(**args)
        datasets.append(dataset)

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return dataset


def make_train_data_loader(datasets, ims_per_gpu, train_sampler):
    num_workers = cfg.TRAIN.LOADER_THREADS
    data_loader = torch.utils.data.DataLoader(
        datasets,
        batch_size=ims_per_gpu,
        shuffle=True if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    return data_loader


def make_test_data_loader(datasets, start_ind, end_ind, is_distributed=True):
    ims_per_gpu = cfg.TEST.IMS_PER_GPU
    if start_ind == -1 or end_ind == -1:
        test_sampler = torch.utils.data.distributed.DistributedSampler(datasets) if is_distributed else None
    else:
        test_sampler = samplers.RangeSampler(start_ind, end_ind)
    num_workers = cfg.TEST.LOADER_THREADS
    data_loader = torch.utils.data.DataLoader(
        datasets,
        batch_size=ims_per_gpu,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    return data_loader
