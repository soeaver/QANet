import datetime
import json
import logging
import numpy as np
import os
from collections import defaultdict
from contextlib import contextmanager

import torch

from lib.utils.misc import logging_rank, setup_logging

_CURRENT_STORAGE_STACK = []


def get_event_storage():
    assert len(
        _CURRENT_STORAGE_STACK
    ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE_STACK[-1]


class EventWriter:
    """
    Base class for writers that obtain events from :class:`EventStorage` and process them.
    """

    def write(self, **kwargs):
        raise NotImplementedError

    def close(self):
        pass


class JSONWriter(EventWriter):
    def __init__(self, json_file, window_size=20):
        """
        Args:
            json_file (str): path to the json file. New data will be appended if the file exists.
            window_size (int): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True.
        """
        self.file_handle = open(json_file, "a")
        self.window_size = window_size

    def write(self, **kwargs):
        storage = get_event_storage()
        to_save = {"iteration": storage.iter + 1}
        to_save.update(storage.latest_with_smoothing_hint(self.window_size))
        self.file_handle.write(json.dumps(to_save, sort_keys=True) + "\n")
        self.file_handle.flush()
        try:
            os.fsync(self.file_handle.fileno())
        except AttributeError:
            pass

    def close(self):
        self.file_handle.close()


class TensorboardXWriter(EventWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): The directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self.window_size = window_size
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir, **kwargs)

    def write(self, **kwargs):
        storage = get_event_storage()
        for k, v in storage.latest_with_smoothing_hint(self.window_size).items():
            self.writer.add_scalar(k, v, storage.iter)

    def close(self):
        if hasattr(self, "writer"):  # doesn't exist when the code fails at import
            self.writer.close()


class CommonMetricPrinter(EventWriter):
    """
    Print __common__ metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.
    To print something different, please implement a similar printer by yourself.
    """

    def __init__(self, yaml, max_iter):
        """
        Args:
            max_iter (int): the maximum number of iterations to train.
                Used to compute ETA.
        """
        self.max_iter = max_iter
        self.yaml = yaml
        logger = logging.getLogger("Training")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        plain_formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")
        self.logger = setup_logging(self.yaml, logger, local_plain_formatter=plain_formatter)

    def write(self, epoch, max_epoch, **kwargs):
        storage = get_event_storage()
        iteration = storage.iter

        data_time, time, metrics = None, None, {}
        eta_string = "N/A"
        try:
            data_time = storage.history("data_time").avg(20)
            time = storage.history("time").global_avg()
            if max_epoch is not None:
                eta_iter = max_epoch * self.max_iter - iteration - 1
                iteration = iteration % self.max_iter
            else:
                eta_iter = self.max_iter - iteration
            eta_seconds = time * (eta_iter)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            for k, v in storage.latest().items():
                if "acc" in k:
                    metrics[k] = v
        except KeyError:  # they may not exist in the first few iterations (due to warmup)
            pass

        try:
            lr = "{:.6f}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        losses = [
            "{}: {:.4f}".format(k, v.median(20))
            for k, v in storage.histories().items()
            if "loss" in k and "total" not in k
        ]
        skip_losses = len(losses) == 1
        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        lines = """\
|-[{yaml}]-{epoch}[iter: {iter}/{max_iter}]-[lr: {lr}]-[eta: {eta}]
                 |-[{memory}]-[{time}]-[{data_time}] 
                 |-[total loss: {total_loss}]{losses}
\
""".format(
            yaml=self.yaml.split('/')[-1] + '.yaml',
            eta=eta_string,
            iter=iteration + 1,  # start from iter 1
            epoch='' if epoch is None else '[epoch: {}/{}]-'.format(epoch, max_epoch),
            max_iter=self.max_iter,
            lr=lr,
            memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            time="iter_time: {:.4f}".format(time) if time is not None else "iter_time: N/A",
            data_time="data_time: {:.4f}".format(data_time) if data_time is not None else "",
            total_loss="{:.4f}".format(storage.histories()["total_loss"].median(20)),
            losses="-[losses]-[{}]".format("  ".join(losses)) if not skip_losses else "",
        )

        if len(metrics):
            lines += """\
                 {metrics}\
""".format(metrics="|" + "".join(
                ["-[{}: {:.4f}]".format(k, v) for k, v in metrics.items()]
            )
           )
        else:
            lines = lines[:-1]
        logging_rank(lines, self.logger)


class EventStorage:
    """
    The user-facing class that provides metric storage functionalities.
    In the future we may add support for storing / logging other types of data if needed.
    """

    def __init__(self, start_iter=0, log_period=20, iter_per_epoch=-1):
        """
        Args:
            start_iter (int): the iteration number to start with
        """
        self.window_size = iter_per_epoch if iter_per_epoch != -1 else log_period
        self._history = defaultdict(HistoryBuffer)
        self._smoothing_hints = {}
        self._latest_scalars = {}
        self._iter = start_iter
        self._current_prefix = ""

    def put_scalar(self, name, value, smoothing_hint=True):
        """
        Add a scalar `value` to the `HistoryBuffer` associated with `name`.
        Args:
            smoothing_hint (bool): a 'hint' on whether this scalar is noisy and should be
                smoothed when logged. The hint will be accessible through
                :meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
                and apply custom smoothing rule.
                It defaults to True because most scalars we save need to be smoothed to
                provide any useful signal.
        """
        name = self._current_prefix + name
        history = self._history[name]
        value = float(value)
        history.update(value, self._iter)
        self._latest_scalars[name] = value

        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None:
            assert (
                    existing_hint == smoothing_hint
            ), "Scalar {} was put with a different smoothing_hint!".format(name)
        else:
            self._smoothing_hints[name] = smoothing_hint

    def put_scalars(self, *, smoothing_hint=True, **kwargs):
        """
        Put multiple scalars from keyword arguments.
        Examples:
            storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
        """
        for k, v in kwargs.items():
            self.put_scalar(k, v, smoothing_hint=smoothing_hint)

    def history(self, name):
        """
        Returns:
            HistoryBuffer: the scalar history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {}!".format(name))
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self._history

    def latest(self):
        """
        Returns:
            dict[name -> number]: the scalars that's added in the current iteration.
        """
        return self._latest_scalars

    def latest_with_smoothing_hint(self, window_size=20):
        """
        Similar to :meth:`latest`, but the returned values
        are either the un-smoothed original latest value,
        or a median of the given window_size,
        depend on whether the smoothing_hint is True.
        This provides a default behavior that other writers can use.
        """
        result = {}

        for k, v in self._latest_scalars.items():
            result[k] = self._history[k].median(window_size) if self._smoothing_hints[k] else v
        return result

    def smoothing_hints(self):
        """
        Returns:
            dict[name -> bool]: the user-provided hint on whether the scalar
                is noisy and needs smoothing.
        """
        return self._smoothing_hints

    def step(self):
        """
        User should call this function at the beginning of each iteration, to
        notify the storage of the start of a new iteration.
        The storage will then be able to associate the new data with the
        correct iteration number.
        """
        self._iter += 1
        self._latest_scalars = {}

    @property
    def iter(self):
        return self._iter

    @property
    def iteration(self):
        # for backward compatibility
        return self._iter

    def __enter__(self):
        _CURRENT_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_STORAGE_STACK[-1] == self
        _CURRENT_STORAGE_STACK.pop()

    @contextmanager
    def name_scope(self, name):
        """
        Yields:
            A context within which all the events added to this storage
            will be prefixed by the name scope.
        """
        old_prefix = self._current_prefix
        self._current_prefix = name.rstrip("/") + "/"
        yield
        self._current_prefix = old_prefix


class HistoryBuffer:
    """
    Track a series of scalar values and provide access to smoothed values over a
    window or the global average of the series.
    """

    def __init__(self, max_length: int = 1000000):
        """
        Args:
            max_length: maximal number of values that can be stored in the
                buffer. When the capacity of the buffer is exhausted, old
                values will be removed.
        """
        self._max_length = max_length  # int
        self._data = []  # List[Tuple[float, float]]  (value, iteration) pairs
        self._count = 0  # int
        self._global_avg = 0  # float

    def update(self, value: float, iteration: float = None):
        """
        Add a new scalar value produced at certain iteration. If the length
        of the buffer exceeds self._max_length, the oldest element will be
        removed from the buffer.
        """
        if iteration is None:
            iteration = self._count
        if len(self._data) == self._max_length:
            self._data.pop(0)
        self._data.append((value, iteration))

        self._count += 1
        self._global_avg += (value - self._global_avg) / self._count

    def latest(self):
        """
        Return the latest scalar value added to the buffer.
        """
        return self._data[-1][0]

    def median(self, window_size: int):
        """
        Return the median of the latest `window_size` values in the buffer.
        """
        return np.median([x[0] for x in self._data[-window_size:]])

    def avg(self, window_size: int):
        """
        Return the mean of the latest `window_size` values in the buffer.
        """
        return np.mean([x[0] for x in self._data[-window_size:]])

    def global_avg(self):
        """
        Return the mean of all the elements in the buffer. Note that this
        includes those getting removed due to limited buffer storage.
        """
        return self._global_avg

    def values(self):
        """
        Returns:
            list[(number, iteration)]: content of the current buffer.
        """
        return self._data
