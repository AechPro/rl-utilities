"""
File: welford_running_stats.py
Author: Matthew Allen

Description:
    An implementation of Welford's algorithm for running statistics.
"""

import json
from typing import Union, Any, Tuple
from rl_utilities.running_stats import RunningStatistics


class WelfordRunningStat(RunningStatistics):
    """
    Welford's algorithm for running statistics.
    https://www.johndcook.com/blog/skewness_kurtosis/
    """

    def __init__(self, shape: Union[Tuple[int, ...], int], backend: str = RunningStatistics.PYTHON_BACKEND) -> None:
        self.running_mean = None
        self.running_variance = None
        super().__init__(shape, backend)

    def _init_statistics(self) -> None:
        """Initialize Welford-specific statistics."""
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
        current_count = self.count
        self.count += 1

        # In the specific case that we're tracking a list of python floats, we need to do it this way.
        if self._is_python_list_backend():
            for i in range(len(self.running_mean)):
                delta = sample[i] - self.running_mean[i]
                delta_n = delta / self.count
                self.running_mean[i] += delta_n
                self.running_variance[i] += delta * delta_n * current_count
        else:
            delta = sample - self.running_mean
            delta_n = delta / self.count
            self.running_mean += delta_n
            self.running_variance += delta * delta_n * current_count

    def reset(self) -> None:
        """Reset all statistics."""
        del self.running_mean
        del self.running_variance
        self.__init__(self.shape, self._backend)

    @property
    def mean(self) -> Any:
        if self.count < 2:
            return self.zeros
        return self.running_mean

    @property
    def var(self) -> Any:
        if self.count < 2:
            return self.ones
        if self._backend == WelfordRunningStat.PYTHON_BACKEND:
            # Scalar
            if not hasattr(self.running_variance, "__len__"):
                return self.running_variance / (self.count - 1)
            # Vector or nested
            def div(x):
                if isinstance(x, list):
                    return [div(xx) for xx in x]
                return x / (self.count - 1)
            return div(self.running_variance)
        else:
            return self.running_variance / (self.count - 1)

    def to_json(self) -> str:
        """Convert statistics to JSON string."""
        data = {
            'shape': self.shape,
            'backend': self._backend,
            'count': self.count,
            'running_mean': self._prepare_data_for_json(self.running_mean),
            'running_variance': self._prepare_data_for_json(self.running_variance)
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'WelfordRunningStat':
        """Create WelfordRunningStat instance from JSON string."""
        data = json.loads(json_str)
        # Create new instance
        welford = cls(data['shape'], data['backend'])
        welford.count = data['count']
        
        # Restore running statistics
        welford.running_mean = welford._restore_data_from_json(data['running_mean'])
        welford.running_variance = welford._restore_data_from_json(data['running_variance'])
        
        return welford
