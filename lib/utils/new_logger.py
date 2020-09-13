import os
import time
import shutil
import logging
import datetime
import itertools
import numpy as np
from collections import OrderedDict, Counter

import torch

from .timer import Timer
from .events import EventStorage, EventWriter, CommonMetricPrinter, JSONWriter, TensorboardXWriter
from .misc import logging_rank
from .comm import is_main_process, gather
from .net import get_bn_modules, update_bn_stats


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


class TrainHook:
    def before_train(self, **kwargs):
        pass

    def after_train(self, **kwargs):
        pass

    def before_step(self, **kwargs):
        pass

    def after_step(self, **kwargs):
        pass


class TestHook():
    """Track vital testing statistics."""

    def __init__(self, cfg_filename, logperiod=10, num_warmup=5):
        self.cfg_filename = cfg_filename
        self.logperiod = logperiod
        self.num_warmup = num_warmup
        self.timers = OrderedDict()
        self.iter = 0

        self.default_timers = ('iter', 'data', 'infer', 'post')
        for name in self.default_timers:
            self.add_timer(name)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name.endswith('_tic'):
            return lambda : self.tic(name[:-4])
        elif name.endswith('_toc'):
            return lambda : self.toc(name[:-4])
        else:
            raise AttributeError(name)

    def wait(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def add_timer(self, name):
        if name in self.timers:
            raise ValueError(
                "Trying to add a existed Timer which is named '{}'!".format(name)
            )
        timer = Timer()
        self.timers[name] = timer
        return timer

    def reset_timer(self):
        for _, timer in self.timers:
            timer.reset()

    def tic(self, name):
        if name == 'iter':
            self.iter += 1
        timer = self.timers.get(name, None)
        if not timer:
            timer = self.add_timer(name)
        timer.tic()

    def toc(self, name):
        timer = self.timers.get(name, None)
        if not timer:
            raise ValueError(
                "Trying to toc a non-existent Timer which is named '{}'!".format(name)
            )
        if self.iter > self.num_warmup:
            self.wait()
            return timer.toc(average=False)

    def log_stats(self, cur_idx, start_ind, end_ind, total_num_images, ims_per_gpu=1, suffix=None, log_all=False):
        """Log the tracked statistics."""
        if (cur_idx + 1) % self.logperiod == 0 or cur_idx == end_ind - 1:
            eta_seconds = self.timers['iter'].average_time / ims_per_gpu * (end_ind - cur_idx - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            lines = ['[Testing][range:{}-{} of {}][{}/{}]'. \
                         format(start_ind + 1, end_ind, total_num_images, cur_idx + 1, end_ind),
                     '[{:.3f}s = {:.3f}s + {:.3f}s + {:.3f}s][eta: {}]'. \
                         format(*[self.timers[name].average_time / ims_per_gpu for name in self.default_timers], eta)]

            if log_all:
                lines.append('\n|')
                for name, timer in self.timers.items():
                    if name not in self.default_timers:
                        lines.append('{}: {:.3f}s|'.format(name, timer.average_time / ims_per_gpu))
            if suffix is not None:
                lines.append(suffix)
            logging_rank(''.join(lines))


class PeriodicWriter(TrainHook):
    """
    Write events to EventStorage periodically.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def __init__(self, cfg, writers, max_iter):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            period (int):
        """
        self.cfg = cfg
        self.writers = writers
        self.max_iter = max_iter
        for w in writers:
            assert isinstance(w, EventWriter), w

    def after_step(self, storage, epoch=None, **kwargs):
        if epoch is not None:
            max_epoch = self.cfg.SOLVER.MAX_EPOCHS
            iter = storage.iter % self.max_iter
        else:
            max_epoch = None
            iter = storage.iter

        if epoch is not None:
            storage.put_scalar("epoch", epoch, smoothing_hint=False)
        if (iter + 1) % self.cfg.DISPLAY_ITER == 0 or (
                iter == self.max_iter - 1
        ):
            for writer in self.writers:
                writer.write(epoch=epoch, max_epoch=max_epoch)

    def after_train(self, **kwargs):
        for writer in self.writers:
            writer.close()


class IterationTimer(TrainHook):
    def __init__(self, max_iter, start_iter, warmup_iter, ignore_warmup_time):
        self.warmup_iter = warmup_iter
        self.step_timer = Timer()
        self.start_iter = start_iter
        self.max_iter = max_iter
        self.ignore_warmup_time = ignore_warmup_time

    def before_train(self, **kwargs):
        self.start_time = time.perf_counter()
        self.total_timer = Timer()
        self.total_timer.pause()

    def after_train(self, storage, **kwargs):
        iter = storage.iter
        total_time = time.perf_counter() - self.start_time
        total_time_minus_hooks = self.total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = iter + 1 - self.start_iter - self.warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logging_rank(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logging_rank(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self, **kwargs):
        self.step_timer.reset()
        self.total_timer.resume()

    def after_step(self, storage, **kwargs):
        # +1 because we're in after_step
        if self.ignore_warmup_time:
            # ignore warm up time cost
            if storage.iter >= self.warmup_iter:
                sec = self.step_timer.seconds()
                storage.put_scalars(time=sec)
            else:
                self.start_time = time.perf_counter()
                self.total_timer.reset()
        else:
            sec = self.step_timer.seconds()
            storage.put_scalars(time=sec)

        self.total_timer.pause()


class LRScheduler(TrainHook):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer, scheduler):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim._LRScheduler)
        """
        self.optimizer = optimizer
        self.scheduler = scheduler

        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    self._best_param_group_id = i
                    break
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    self._best_param_group_id = i
                    break

    def after_step(self, storage, **kwargs):
        lr = self.optimizer.param_groups[self._best_param_group_id]["lr"]
        storage.put_scalar("lr", lr, smoothing_hint=False)
        self.scheduler.step()


class PreciseBN(TrainHook):
    """
    The standard implementation of BatchNorm uses EMA in inference, which is
    sometimes suboptimal.
    This class computes the true average of statistics rather than the moving average,
    and put true averages to every BN layer in the given model.
    It is executed every ``period`` iterations and after the last iteration.
    """

    def __init__(self, precise_bn_args, period, num_iter, max_iter):
        if len(get_bn_modules(precise_bn_args[1])) == 0:
            logging_rank(
                "PreciseBN is disabled because model does not contain BN layers in training mode."
            )
            self.disabled = True
            return

        self.data_loader = precise_bn_args[0]
        self.model = precise_bn_args[1]
        self.device = precise_bn_args[2]
        self.num_iter = num_iter
        self.period = period
        self.max_iter = max_iter
        self.disabled = False

        self.data_iter = None

    def after_step(self, storage, epoch=None, **kwargs):
        if epoch is not None:
            next_iter = storage.iter % self.max_iter + 1
            is_final = next_iter == self.max_iter and epoch == kwargs.pop('max_epochs')
        else:
            next_iter = storage.iter + 1
            is_final = next_iter == self.max_iter
        if is_final or (self.period > 0 and next_iter % self.period == 0):
            self.update_stats()

    def update_stats(self):
        """
        Update the model with precise statistics. Users can manually call this method.
        """
        if self.disabled:
            return

        if self.data_iter is None:
            self.data_iter = iter(self.data_loader)

        def data_loader():
            for num_iter in itertools.count(1):
                if num_iter % 100 == 0:
                    logging_rank(
                        "Running precise-BN ... {}/{} iterations.".format(num_iter, self.num_iter)
                    )
                # This way we can reuse the same iterator
                yield next(self.data_iter)

        with EventStorage():
            logging_rank(
                    "Running precise-BN for {} iterations...  ".format(self.num_iter)
                    + "Note that this could produce different statistics every time."
                )
            update_bn_stats(self.model, data_loader(), self.device, self.num_iter)


def build_train_hooks(cfg, optimizer, scheduler, max_iter, warmup_iter, ignore_warmup_time=False,
                      precise_bn_args=None):
    """
    Build a list of default train hooks.
    """
    start_iter = scheduler.iteration

    ret = [
        IterationTimer(max_iter, start_iter, warmup_iter, ignore_warmup_time),
        LRScheduler(optimizer, scheduler),
    ]

    if cfg.TEST.PRECISE_BN.ENABLED:
        if get_bn_modules(precise_bn_args[1]):
            ret.append(PreciseBN(
                precise_bn_args, cfg.TEST.PRECISE_BN.PERIOD, cfg.TEST.PRECISE_BN.NUM_ITER, max_iter
            ))

    if is_main_process():
        write_ret = [CommonMetricPrinter(cfg.CKPT, max_iter)]

        if cfg.TRAIN.SAVE_AS_JSON:
            write_ret.append(JSONWriter(os.path.join(cfg.CKPT, "metrics.json")))
        if cfg.TRAIN.USE_TENSORBOARD:
            log_dir = os.path.join(cfg.CKPT, "tensorboard_log")
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.mkdir(log_dir)
            write_ret.append(TensorboardXWriter(log_dir))
        ret.append(PeriodicWriter(cfg, write_ret, max_iter))

    return ret


def build_test_hooks(cfg_filename, log_period, num_warmup=4):
    """
    Build default test hooks.
    """
    assert log_period > num_warmup
    return TestHook(cfg_filename, log_period, num_warmup)


def write_metrics(metrics_dict, storage):
    metrics_dict = {
        k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
        for k, v in metrics_dict.items()
    }
    # gather metrics among all workers for logging
    all_metrics_dict = gather(metrics_dict)

    if is_main_process():
        max_keys = ("data_time", "best_acc1")
        for m_k in max_keys:
            if m_k in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                m_v = np.max([x.pop(m_k) for x in all_metrics_dict])
                storage.put_scalar(m_k, m_v)

        # average the rest metrics
        metrics_dict = {
            k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
        }
        total_losses_reduced = sum(v if 'loss' in k else 0 for k, v in metrics_dict.items())

        storage.put_scalar("total_loss", total_losses_reduced)
        if len(metrics_dict) >= 1:
            storage.put_scalars(**metrics_dict)
