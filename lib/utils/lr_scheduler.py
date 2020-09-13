import math
import numpy as np
from bisect import bisect_right

from torch.optim import lr_scheduler

from .misc import logging_rank


def _get_lr_change_ratio(cur_lr, new_lr):
    eps = 1e-10
    ratio = np.max((new_lr / np.max((cur_lr, eps)), cur_lr / np.max((new_lr, eps))))
    return ratio


class LearningRateScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, solver, start_iter=0, iter_per_epoch=-1, last_epoch=-1):
        self.solver = solver
        self.iteration = start_iter
        if 'MAX_ITER' in self.solver:
            self.milestones = self.solver.STEPS
            self.max_iter = self.solver.MAX_ITER
            self.warmup_iters = self.solver.WARM_UP_ITERS
        else:
            self.iter_per_epoch = iter_per_epoch
            self.conver_epoch2iter()

        assert self.solver.LR_POLICY in ['STEP', 'COSINE', 'STEP_COSINE', 'POLY']
        assert self.solver.WARM_UP_METHOD in ['CONSTANT', 'LINEAR']
        assert list(self.milestones) == sorted(self.milestones)
        self.gamma = self.solver.GAMMA
        self.warmup_factor = self.solver.WARM_UP_FACTOR
        self.warmup_method = self.solver.WARM_UP_METHOD
        self.lr_factor = 0
        self.info = dict(best_acc=0.0, best_epoch=1, cur_acc=0.0, cur_epoch=1)
        super().__init__(optimizer, last_epoch)

    def conver_epoch2iter(self):
        """Convert the epoch style parameters to corresponding iteration.
        """
        self.max_iter = self.solver.MAX_EPOCHS * self.iter_per_epoch
        self.warmup_iters = self.solver.WARM_UP_EPOCH * self.iter_per_epoch
        self.milestones = [epoch * self.iter_per_epoch for epoch in self.solver.STEPS]  # only useful for step policy

    def get_lr(self):
        """Update learning rate
        """
        warmup_factor = self.get_warmup_factor(
            self.warmup_method, self.iteration, self.warmup_iters, self.warmup_factor
        )
        if self.solver.LR_POLICY == "STEP":
            lr_factor = self.get_step_factor(warmup_factor)
        elif self.solver.LR_POLICY == "COSINE":
            lr_factor = self.get_cosine_factor(warmup_factor)
        elif self.solver.LR_POLICY == 'STEP_COSINE':
            if self.iteration < self.milestones[-1]:
                lr_factor = self.get_step_factor(warmup_factor)
            else:
                lr_factor = self.get_cosine_lrs(warmup_factor)
        elif self.solver.LR_POLICY == 'POLY':
            lr_factor = self.get_poly_factor(warmup_factor)
        else:
            raise KeyError('Unknown SOLVER.LR_POLICY: {}'.format(self.solver.LR_POLICY))

        ratio = _get_lr_change_ratio(lr_factor, self.lr_factor)
        if self.lr_factor != lr_factor and ratio > self.solver.LOG_LR_CHANGE_THRESHOLD:
            if lr_factor * self.solver.BASE_LR > 1e-7 and self.iteration > 1:
                logging_rank('Changing learning rate {:.6f} -> {:.6f}'.format(
                    self.lr_factor * self.solver.BASE_LR, lr_factor * self.solver.BASE_LR)
                )
        self.lr_factor = lr_factor

        self.iteration += 1

        return [lr_factor * base_lr for base_lr in self.base_lrs]

    def get_step_factor(self, warmup_factor):
        """Get learning rate factor when using 'STEP' policy
        """
        return warmup_factor * self.gamma ** bisect_right(self.milestones, self.iteration)

    def get_cosine_factor(self, warmup_factor):
        """Get learning rate factor when using 'COSINE' policy
        """
        return warmup_factor * 0.5 * (1.0 + math.cos(math.pi * self.iteration / self.max_iter))

    def get_poly_factor(self, warmup_factor):
        """Get learning rate factor when using 'POLY' policy
        """
        return warmup_factor * (1. - float(self.iteration / self.max_iter)) ** self.solver.LR_POW

    def _compute_values(self):
        # The new interface
        return self.get_lr()

    def get_warmup_factor(self, method, iter, warmup_iters, warmup_factor):
        if iter >= warmup_iters:
            return 1.0
        if method == "CONSTANT":
            return warmup_factor
        elif method == "LINEAR":
            alpha = iter / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        else:
            raise ValueError("Unknown warmup method: {}".format(method))

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
