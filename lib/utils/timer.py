import time
from typing import Optional
from collections import OrderedDict

import torch

from .misc import logging_rank


class Timer:
    """
    A timer which computes the time elapsed since the start/reset of the timer.
    """

    def __init__(self):
        self.reset()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def reset(self):
        """
        Reset the timer.
        """
        self._start = time.perf_counter()
        self._paused = None  # Optional[float]
        self._total_paused = 0
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def pause(self):
        """
        Pause the timer.
        """
        if self._paused is not None:
            raise ValueError("Trying to pause a Timer that is already paused!")
        self._paused = time.perf_counter()

    def is_paused(self):
        """
        Returns:
            bool: whether the timer is currently paused
        """
        return self._paused is not None

    def resume(self):
        """
        Resume the timer.
        """
        if self._paused is None:
            raise ValueError("Trying to resume a Timer that is not paused!")
        self._total_paused += time.perf_counter() - self._paused
        self._paused = None

    def seconds(self):
        """
        Returns:
            (float): the total number of seconds since the start/reset of the
                timer, excluding the time when the timer is paused.
        """
        if self._paused is not None:
            end_time = self._paused  # float / type: ignore
        else:
            end_time = time.perf_counter()
        return end_time - self._start - self._total_paused


class _DebugTimer(object):
    '''
    Track vital debug statistics.

    Usage:
        1. from pet.utils.timer import debug_timer

        2. with debug_timer('timer1'):
               code1

           debug_timer.tic('timer2')
           code2
           debug_timer.toc('timer2')

           debug_timer.timer3_tic()
           code3
           debug_timer.timer3_toc()

        3. debug_timer.log()
    
    TODO: multithreading support
    '''
    __TIMER__ = None
    def __new__(cls, *args, **kwargs):
        if cls.__TIMER__ is None:
            cls.__TIMER__ = super().__new__(cls)
        return cls.__TIMER__

    def __init__(self, num_warmup=5):
        super(_DebugTimer, self).__init__()
        self.num_warmup = num_warmup
        self.timers = OrderedDict()
        self.context_stacks = []
        self.calls = 0

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name.endswith('_tic'):
            return lambda : self.tic(name[:-4])
        elif name.endswith('_toc'):
            return lambda : self.toc(name[:-4])
        else:
            raise AttributeError(name)

    def __call__(self, name):
        self.context_stacks.append(name)
        return self

    def __enter__(self):
        name = self.context_stacks[-1]
        self.tic(name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        name = self.context_stacks.pop(-1)
        self.toc(name)
        if exc_type:
            print(exc_type, exc_value)
            print(traceback)

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
        if self.calls >= self.num_warmup:
            self.wait()
            return timer.toc(average=False)

    def log(self, logperiod=10):
        """
        Log the tracked statistics.
        Eg.: | timer1: xxxs | timer2: xxxms | timer3: xxxms |
        """
        self.calls += 1
        if self.calls % logperiod == 0 and self.timers:
            lines = ['']
            for name, timer in self.timers.items():
                avg_time = timer.average_time
                suffix = 's'
                if avg_time < 0.01:
                    avg_time *= 1000
                    suffix = 'ms'
                lines.append(' {}: {:.3f}{} '.format(name, avg_time, suffix))
            lines.append('')
            logging_rank('|'.join(lines))


debug_timer = _DebugTimer()
