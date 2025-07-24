"""
File: exponential_moving_average.py
Author: Matthew Allen

Description:
    An implementation of exponential moving average for running statistics.
    Supports tracking both mean and variance using exponential weights.
"""

import json
from typing import Union, Any, Tuple
from rl_utilities.running_stats import RunningStatistics


class ExponentialMovingAverage(RunningStatistics):
    
    def __init__(self, shape: Union[Tuple[int, ...], int], backend: str = RunningStatistics.PYTHON_BACKEND, alpha: float = 0.1) -> None:
        """
        Initialize exponential moving average tracker.
        
        :param shape: Shape of the data to track
        :param backend: Backend to use ('python', 'numpy', 'torch')
        :param alpha: Smoothing factor in [0, 1]. Higher values give more weight to recent observations.
        """
        self.running_mean = None
        self.running_variance = None
        self.alpha = alpha
        super().__init__(shape, backend)

    def _init_statistics(self) -> None:
        """Initialize EMA-specific statistics."""
        if self._backend == self.NUMPY_BACKEND:
            import numpy as np
            self.running_mean = np.zeros(self.shape, dtype=np.float32)
            self.running_variance = np.zeros(self.shape, dtype=np.float32)
        elif self._backend == self.TORCH_BACKEND:
            import torch
            self.running_mean = torch.zeros(self.shape, dtype=torch.float32)
            self.running_variance = torch.zeros(self.shape, dtype=torch.float32)
        else:
            # Handle shape=() (scalar), shape=(N,) (vector), shape=(N,M,...) (matrix)
            if self.shape == () or self.shape == [] or self.shape == 0:
                self.running_mean = 0.0
                self.running_variance = 0.0
            elif isinstance(self.shape, tuple) and len(self.shape) == 1:
                self.running_mean = [0.0] * self.shape[0]
                self.running_variance = [0.0] * self.shape[0]
            else:
                # Multi-dim: create nested lists
                def make_nested(val, shape):
                    if len(shape) == 1:
                        return [val] * shape[0]
                    return [make_nested(val, shape[1:]) for _ in range(shape[0])]
                self.running_mean = make_nested(0.0, self.shape)
                self.running_variance = make_nested(0.0, self.shape)



    def update(self, sample: Any) -> None:
        self.count += 1
        
        # For the first sample, initialize with the sample value
        if self.count == 1:
            if self._is_python_list_backend():
                for i in range(len(self.running_mean)):
                    self.running_mean[i] = sample[i]
                    self.running_variance[i] = 0.0
            else:
                self.running_mean = self._create_copy(sample)
                self.running_variance = self._create_zeros_like(sample)
            return
        
        if self._is_python_list_backend():
            for i in range(len(self.running_mean)):
                delta = sample[i] - self.running_mean[i]
                self.running_mean[i] = (1 - self.alpha) * self.running_mean[i] + self.alpha * sample[i]
                self.running_variance[i] = (1 - self.alpha) * self.running_variance[i] + self.alpha * (delta * delta)
        else:
            delta = sample - self.running_mean
            self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * sample
            self.running_variance = (1 - self.alpha) * self.running_variance + self.alpha * (delta * delta)

    def reset(self) -> None:
        """Reset all statistics."""
        del self.running_mean
        del self.running_variance
        self.__init__(self.shape, self._backend, self.alpha)

    @property
    def mean(self) -> Any:
        if self.count == 0:
            return self.zeros
        return self.running_mean

    @property
    def var(self) -> Any:
        if self.count == 0:
            return self.zeros
        return self.running_variance

    def to_json(self) -> str:
        """Convert statistics to JSON string."""
        data = {
            'shape': self.shape,
            'backend': self._backend,
            'alpha': self.alpha,
            'count': self.count,
            'running_mean': self._prepare_data_for_json(self.running_mean),
            'running_variance': self._prepare_data_for_json(self.running_variance)
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'ExponentialMovingAverage':
        """Create ExponentialMovingAverage instance from JSON string."""
        data = json.loads(json_str)
        
        # Create new instance
        ema = cls(data['shape'], data['backend'], data['alpha'])
        ema.count = data['count']
        
        # Restore running statistics
        ema.running_mean = ema._restore_data_from_json(data['running_mean'])
        ema.running_variance = ema._restore_data_from_json(data['running_variance'])
        
        return ema


