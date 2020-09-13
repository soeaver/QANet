import math
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

from lib.utils.misc import logging_rank


class RepeatFactorInstanceTrainingSampler(Sampler):
    """
    Similar to DistributedSampler, but suitable for training on class imbalanced datasets
    like LVIS, OID. In each epoch, an image may appear multiple times based on its "repeat
    factor". The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1] is defined
    as the fraction of images in the training set (without repeats) in which category c
    appears.

    See https://arxiv.org/abs/1908.03195 (>= v2) Appendix B.2.
    """

    def __init__(self, dataset, config, num_replicas=None, rank=None, shuffle=True):
        """
        Args:
            dataset: COCODataset.
            config:
                REPEAT_THRESHOLD (float): frequency used for control imgs per epoch
                MAX_REPEAT_TIMES (float) : max repeat times for single epoch
                MIN_REPEAT_TIMES (float) : min repeat times for single epoch
                POW(float): 0.5 for lvis paper sqrt ,1.0 for linear
            shuffle (bool): whether to shuffle the indices or not
        """
        self.shuffle = shuffle
        self.config = config
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        # Get annotations list
        coco_json = dataset.coco
        ids = dataset.ids
        dataset_dict_img = []
        for idx, img_id in enumerate(ids):
            dataset_dict_img.append({"annotations": coco_json.loadAnns(dataset.ann_ids[idx])[0]})

        # Get fractional repeat factors and split into whole number (_int_part)
        # and fractional (_frac_part) parts.
        rep_factors = self._get_repeat_factors(dataset_dict_img)
        self._int_part = torch.trunc(rep_factors)
        self._frac_part = rep_factors - self._int_part

    def _get_repeat_factors(self, dataset_dicts):
        """
        Compute (fractional) per-image repeat factors.

        Args:
            dataset_dicts (list) : per-image annotations

        Returns:
            torch.Tensor: the i-th element is the repeat factor for the dataset_dicts image
                at index i.
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_id = dataset_dict["annotations"]["category_id"]
            category_freq[cat_id] += 1
        num_images = len(dataset_dicts)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images
        # 2. For each category c, compute the category-level repeat factor:
        #    lvis paper: r(c) = max(1, sqrt(t / f(c)))
        #    common: r(c) = max(i, min(a,pow(t / f(c),alpha)))
        # category_rep = {
        #     cat_id: max(self.config.MIN_REPEAT_TIMES, min(self.config.MAX_REPEAT_TIMES, math.pow(
        #         (self.config.REPEAT_THRESHOLD / cat_freq), self.config.POW)))
        #     for cat_id, cat_freq in category_freq.items()
        # }
        category_rep = {
            cat_id: max(
                self.config.MIN_REPEAT_TIMES,
                math.pow((self.config.REPEAT_THRESHOLD / cat_freq), self.config.POW)
            )
            for cat_id, cat_freq in category_freq.items()
        }
        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for dataset_dict in dataset_dicts:
            rep_factors.append(category_rep[dataset_dict["annotations"]["category_id"]])
        logging_rank('max(rep_factors): {} , min(rep_factors): {} , len(rep_factors): {}'.
                     format(max(rep_factors), min(rep_factors), len(rep_factors)),
                     )

        return torch.tensor(rep_factors, dtype=torch.float32)

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.

        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = self._get_epoch_indices(g)
            randperm = torch.randperm(len(indices), generator=g).tolist()
            indices = indices[randperm]
        else:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = self._get_epoch_indices(g)
            # indices = torch.arange(len(self.dataset)).tolist()

        # when balance len(indices) diff from dataset image_num
        self.total_size = len(indices)
        logging_rank('balance sample total_size: {}'.format(self.total_size))
        # subsample
        self.num_samples = int(len(indices) / self.num_replicas)
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = self._get_epoch_indices(g)
        return int(len(indices) / self.num_replicas)

    def set_epoch(self, epoch):
        self.epoch = epoch
