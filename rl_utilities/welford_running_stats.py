"""
File: running_stats.py
Author: Matthew Allen

Description:
    An implementation of Welford's algorithm for running statistics.
"""

import os
import json
from typing import Union, Any, List, Tuple, Dict


class WelfordRunningStat(object):
    PYTHON_BACKEND = "python"
    NUMPY_BACKEND = "numpy"
    TORCH_BACKEND = "torch"
    
    """
    https://www.johndcook.com/blog/skewness_kurtosis/
    """

    def __init__(self, shape: Union[Tuple[int, ...], int], backend: str = PYTHON_BACKEND) -> None:
        self.ones = None
        self.zeros = None

        self.running_mean = None
        self.running_variance = None

        self.count = 0
        self.shape = shape
        self._backend = backend
        self._math_lib = None

        self._init_backend()

    def _init_backend(self) -> None:
        if self._backend == WelfordRunningStat.NUMPY_BACKEND:
            import numpy as np
            self._math_lib = np
            self.ones = np.ones(self.shape, dtype=np.float32)
            self.zeros = np.zeros(self.shape, dtype=np.float32)
            self.running_mean = np.zeros(self.shape, dtype=np.float32)
            self.running_variance = np.zeros(self.shape, dtype=np.float32)
        elif self._backend == WelfordRunningStat.TORCH_BACKEND:
            import torch
            self._math_lib = torch
            self.ones = torch.ones(self.shape, dtype=torch.float32)
            self.zeros = torch.zeros(self.shape, dtype=torch.float32)
            self.running_mean = torch.zeros(self.shape, dtype=torch.float32)
            self.running_variance = torch.zeros(self.shape, dtype=torch.float32)
        else:
            import math
            self._math_lib = math
            # Handle shape=() (scalar), shape=(N,) (vector), shape=(N,M,...) (matrix)
            if self.shape == () or self.shape == [] or self.shape == 0:
                self.ones = 1.0
                self.zeros = 0.0
                self.running_mean = 0.0
                self.running_variance = 0.0
            elif isinstance(self.shape, tuple) and len(self.shape) == 1:
                self.ones = [1.0] * self.shape[0]
                self.zeros = [0.0] * self.shape[0]
                self.running_mean = [0.0] * self.shape[0]
                self.running_variance = [0.0] * self.shape[0]
            else:
                # Multi-dim: create nested lists
                def make_nested(val, shape):
                    if len(shape) == 1:
                        return [val] * shape[0]
                    return [make_nested(val, shape[1:]) for _ in range(shape[0])]
                self.ones = make_nested(1.0, self.shape)
                self.zeros = make_nested(0.0, self.shape)
                self.running_mean = make_nested(0.0, self.shape)
                self.running_variance = make_nested(0.0, self.shape)


    def increment(self, samples: Any, num: int) -> None:
        if num > 1:
            for i in range(num):
                self.update(samples[i])
        else:
            self.update(samples)

    def update(self, sample: Any) -> None:
        current_count = self.count
        self.count += 1

        # In the specific case that we're tracking a list of python floats, we need to do it this way.
        if hasattr(self.running_mean, "__len__") and self._backend == WelfordRunningStat.PYTHON_BACKEND:
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
        del self.running_mean
        del self.running_variance

        self.__init__(self.shape, self._backend)

    @property
    def mean(self) -> Any:
        if self.count < 2:
            return self.zeros
        return self.running_mean

    @property
    def std(self) -> Any:
        if self.count < 2:
            return self.ones
        v = self.var
        if self._backend == WelfordRunningStat.PYTHON_BACKEND:
            import math
            def sqrt_nested(x):
                if isinstance(x, list):
                    return [sqrt_nested(xx) for xx in x]
                return math.sqrt(x)
            return sqrt_nested(v)
        return self._math_lib.sqrt(v)

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

    def to_json(self) -> Dict[str, Any]:
        if self._backend == WelfordRunningStat.NUMPY_BACKEND:
            return {"mean":self.running_mean.ravel().tolist(),
                    "var":self.running_variance.ravel().tolist(),
                    "shape":self._math_lib.shape(self.running_mean),
                    "count":self.count}
        elif self._backend == WelfordRunningStat.TORCH_BACKEND:
            return {"mean":self.running_mean.detach().cpu().flatten().tolist(),
                    "var":self.running_variance.detach().cpu().flatten().tolist(),
                    "shape":self._math_lib.shape(self.running_mean),
                    "count":self.count}
        else:
            return {"mean":self.running_mean,
                    "var":self.running_variance,
                    "shape":self.shape,
                    "count":self.count}

    def from_json(self, other_json: Dict[str, Any]) -> None:
        shape = other_json["shape"]
        self.count = other_json["count"]

        if self._backend == WelfordRunningStat.NUMPY_BACKEND:
            self.running_mean = self._math_lib.asarray(other_json["mean"]).reshape(shape)
            self.running_variance = self._math_lib.asarray(other_json["var"]).reshape(shape)
        elif self._backend == WelfordRunningStat.TORCH_BACKEND:
            self.running_mean = self._math_lib.as_tensor(other_json["mean"]).reshape(shape)
            self.running_variance = self._math_lib.as_tensor(other_json["var"]).reshape(shape)
        else:
            self.running_mean = other_json["mean"]
            self.running_variance = other_json["var"]
        print(F"LOADED RUNNING STATS FROM JSON | Mean: {self.running_mean} | Variance: {self.running_variance} | Count: {self.count}")

    def save(self, directory: str) -> None:
        full_path = os.path.join(directory, "RUNNING_STATS.json")
        with open(full_path, 'w') as f:
            json_data = self.to_json()
            json.dump(obj=json_data, fp=f, indent=4)

    def load(self, directory: str) -> None:
        full_path = os.path.join(directory, "RUNNING_STATS.json")
        with open(full_path, 'r') as f:
            json_data = dict(json.load(f))
            self.from_json(json_data)