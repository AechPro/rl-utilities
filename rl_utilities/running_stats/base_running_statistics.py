"""
File: base_running_statistics.py
Author: Matthew Allen

Description:
    Base class for running statistics implementations.
    Handles backend initialization and common functionality.
"""

import os
import json
from typing import Union, Any, List, Tuple, Dict
from abc import ABC, abstractmethod


class RunningStatistics(ABC):
    """Abstract base class for running statistics implementations."""
    
    # Backend constants
    PYTHON_BACKEND = "python"
    NUMPY_BACKEND = "numpy"
    TORCH_BACKEND = "torch"
    
    def __init__(self, shape: Union[Tuple[int, ...], int], backend: str = PYTHON_BACKEND) -> None:
        self.ones = None
        self.zeros = None
        self.count = 0
        self.shape = shape
        self._backend = backend
        self._math_lib = None
        
        self._init_backend()
        self._init_statistics()
    
    def _init_backend(self) -> None:
        """Initialize backend-specific data structures."""
        if self._backend == self.NUMPY_BACKEND:
            import numpy as np
            self._math_lib = np
            self.ones = np.ones(self.shape, dtype=np.float32)
            self.zeros = np.zeros(self.shape, dtype=np.float32)
        elif self._backend == self.TORCH_BACKEND:
            import torch
            self._math_lib = torch
            self.ones = torch.ones(self.shape, dtype=torch.float32)
            self.zeros = torch.zeros(self.shape, dtype=torch.float32)
        else:
            import math
            self._math_lib = math
            # Handle shape=() (scalar), shape=(N,) (vector), shape=(N,M,...) (matrix)
            if self.shape == () or self.shape == [] or self.shape == 0:
                self.ones = 1.0
                self.zeros = 0.0
            elif isinstance(self.shape, tuple) and len(self.shape) == 1:
                self.ones = [1.0] * self.shape[0]
                self.zeros = [0.0] * self.shape[0]
            else:
                # Multi-dim: create nested lists
                def make_nested(val, shape):
                    if len(shape) == 1:
                        return [val] * shape[0]
                    return [make_nested(val, shape[1:]) for _ in range(shape[0])]
                self.ones = make_nested(1.0, self.shape)
                self.zeros = make_nested(0.0, self.shape)
    
    @abstractmethod
    def _init_statistics(self) -> None:
        """Initialize statistics-specific data structures. Must be implemented by subclasses."""
        pass
    
    def _create_zeros_like(self, value: Any) -> Any:
        """Create zeros with same shape and backend as the given value."""
        if self._backend == self.TORCH_BACKEND:
            return self.zeros.clone()
        elif self._backend == self.NUMPY_BACKEND:
            return self.zeros.copy()
        else:
            return self.zeros
    
    def _create_copy(self, value: Any) -> Any:
        """Create a copy of the given value using appropriate backend method."""
        if self._backend == self.TORCH_BACKEND:
            return value.clone()
        elif self._backend == self.NUMPY_BACKEND:
            return value.copy()
        else:
            return value
    
    def _compute_sqrt(self, value: Any) -> Any:
        """Compute square root using appropriate backend."""
        if self._backend == self.PYTHON_BACKEND:
            import math
            if hasattr(value, "__len__"):
                return [math.sqrt(v) for v in value]
            else:
                return math.sqrt(value)
        else:
            return self._math_lib.sqrt(value)
    
    def _is_python_list_backend(self) -> bool:
        """Check if we're using Python backend with list data."""
        return (hasattr(self, 'running_mean') and 
                hasattr(self.running_mean, "__len__") and 
                self._backend == self.PYTHON_BACKEND)
    
    def increment(self, samples: Any, num: int) -> None:
        """Update statistics with multiple samples."""
        if num > 1:
            for i in range(num):
                self.update(samples[i])
        else:
            self.update(samples)
    
    @abstractmethod
    def update(self, sample: Any) -> None:
        """Update statistics with a single sample. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset all statistics. Must be implemented by subclasses."""
        pass
    
    @property
    @abstractmethod
    def mean(self) -> Any:
        """Get the current mean. Must be implemented by subclasses."""
        pass
    
    @property
    @abstractmethod
    def var(self) -> Any:
        """Get the current variance. Must be implemented by subclasses."""
        pass
    
    @property
    def std(self) -> Any:
        """Get the current standard deviation."""
        if self.count == 0:
            return self.ones
        return self._compute_sqrt(self.var)

    def standardize(self, x, safe=True):
        std = self.std
        if self.std is None:
            print("!STANDARDIZATION IS NOT SUPPORTED FOR {} RUNNING STATISTICS!".format(self.__class__.__name__))
            return x

        if safe:
            if self._backend in [RunningStatistics.TORCH_BACKEND, RunningStatistics.NUMPY_BACKEND]:
                std = self._math_lib.where(std == 0, 1, std)
            else:
                if hasattr(std, "__len__"):
                    std = [1 if s == 0 else s for s in std]
                else:
                    std = 1 if std == 0 else std

        return (x - self.mean) / std
    
    def _prepare_data_for_json(self, data: Any) -> Any:
        """Convert tensors/arrays to lists for JSON serialization."""
        if self._backend == self.NUMPY_BACKEND:
            return data.tolist()
        elif self._backend == self.TORCH_BACKEND:
            return data.tolist()
        else:
            return data
    
    def _restore_data_from_json(self, data: Any) -> Any:
        """Restore tensors/arrays from JSON data."""
        if self._backend == self.NUMPY_BACKEND:
            import numpy as np
            return np.array(data, dtype=np.float32)
        elif self._backend == self.TORCH_BACKEND:
            import torch
            return torch.tensor(data, dtype=torch.float32)
        else:
            return data
    
    @abstractmethod
    def to_json(self) -> str:
        """Convert statistics to JSON string. Must be implemented by subclasses."""
        pass
    
    @classmethod
    @abstractmethod
    def from_json(cls, json_str: str) -> 'RunningStatistics':
        """Create instance from JSON string. Must be implemented by subclasses."""
        pass
    
    def save(self, file_path: str) -> None:
        """Save statistics to a file."""
        with open(file_path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, file_path: str) -> 'RunningStatistics':
        """Load statistics from a file."""
        with open(file_path, 'r') as f:
            return cls.from_json(f.read())
