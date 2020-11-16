import os
import torch.utils.data

from lib.data import samplers
from lib.data.collate_batch import BatchCollator
from lib.data.datasets.instance_data import COCOInstanceDataset, COCOInstanceTestDataset
from lib.data.datasets.concat_dataset import ConcatDataset
from lib.utils.comm import get_world_size
from lib.utils.misc import logging_rank

from qanet.datasets.dataset_catalog import contains, get_ann_fn, get_extra_fields, get_im_dir
from qanet.datasets.transform import build_transforms


def build_dataset(cfg, is_train=True):
    dataset_list = cfg.TRAIN.DATASETS if is_train else cfg.TEST.DATASETS
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    for dataset_name in dataset_list:
        assert contains(dataset_name), 'Unknown dataset name: {}'.format(dataset_name)
        assert os.path.exists(get_im_dir(dataset_name)), 'Im dir \'{}\' not found'.format(get_im_dir(dataset_name))
        logging_rank('Creating: {}'.format(dataset_name))

    transforms = build_transforms(cfg, is_train)
    datasets = []
    for dataset_name in dataset_list:
        args = {}
        args['root'] = get_im_dir(dataset_name)
        args['ann_file'] = get_ann_fn(dataset_name)
        args['bbox_file'] = cfg.TEST.INSTANCE_BBOX_FILE
        args['image_thresh'] = cfg.TEST.IMAGE_THRESH
        extra_fields = get_extra_fields(dataset_name)
        ann_types = ()
        if cfg.MODEL.MASK_ON:
            ann_types = ann_types + ('mask',)
            extra_fields.update(dict(mask_format=cfg.DATALOADER.GT_FORMAT.MASK))
        if cfg.MODEL.KEYPOINT_ON:
            ann_types = ann_types + ('keypoints',)
        if cfg.MODEL.PARSING_ON:
            ann_types = ann_types + ('parsing',)
            extra_fields.update(dict(semseg_format=cfg.DATALOADER.GT_FORMAT.SEMSEG))
        if cfg.MODEL.UV_ON:
            ann_types = ann_types + ('uv',)
        args['ann_types'] = ann_types
        args["transforms"] = transforms
        args['extra_fields'] = extra_fields
        if is_train:
            dataset = COCOInstanceDataset(**args)
        else:
            dataset = COCOInstanceTestDataset(**args)
        datasets.append(dataset)

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return dataset


def make_data_sampler(cfg, datasets, shuffle=True):
    if cfg.DATALOADER.SAMPLER_TRAIN == "RepeatFactorInstanceTrainingSampler":
        return samplers.RepeatFactorInstanceTrainingSampler(datasets, cfg.DATALOADER.RFTSAMPLER, shuffle=shuffle)
    else:
        return torch.utils.data.distributed.DistributedSampler(datasets, shuffle=shuffle)


def make_train_data_loader(cfg, datasets, train_sampler):
    num_gpus = get_world_size()
    ims_per_gpu = int(cfg.TRAIN.BATCH_SIZE / num_gpus)
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


def make_test_data_loader(cfg, datasets):
    ims_per_gpu = cfg.TEST.IMS_PER_GPU
    test_sampler = torch.utils.data.distributed.DistributedSampler(datasets)
    num_workers = cfg.TEST.LOADER_THREADS
    collator = BatchCollator(-1)
    data_loader = torch.utils.data.DataLoader(
        datasets,
        batch_size=ims_per_gpu,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=collator,
    )

    return data_loader
