from collections import defaultdict
from typing import Dict, Any, Type, Optional
from rl_utilities.running_stats import RunningStatistics, WelfordRunningStat, ExponentialMovingAverage

class RLDataLogger:
    # Mapping from string identifiers to running statistics classes
    _STATS_CLASSES = {
        'welford': WelfordRunningStat,
        'ema': ExponentialMovingAverage,
        'exponential': ExponentialMovingAverage,  # alias
    }
    
    def __init__(self) -> None:
        """
        RLDataLogger for RL projects. Tracks per-key values and optional running statistics.
        """
        self._log: Dict[str, Any] = {}
        self._stats: Dict[str, RunningStatistics] = {}

    def log(self, **kwargs: Any) -> None:
        """
        Log key-value pairs. Accepts multiple named arguments.
        Example: logger.log(loss=0.5, reward=1.0)
        """
        for key, value in kwargs.items():
            self._log[key] = value

    def log_stats(self, stats_type: Optional[str] = None, **kwargs: Any) -> None:
        """
        Log key-value pairs and update their running statistics (mean, std).
        
        :param stats_type: Optional string identifier for running statistics type:
                          - 'welford': WelfordRunningStat (default)
                          - 'ema' or 'exponential': ExponentialMovingAverage
        :param kwargs: Key-value pairs to log
        
        Example: 
            logger.log_stats(loss=0.5, reward=1.0)  # Uses 'welford'
            logger.log_stats('ema', loss=0.5)  # Uses ExponentialMovingAverage
            logger.log_stats('welford', reward=1.0)  # Uses WelfordRunningStat
        """
        if stats_type is None:
            stats_type = 'ema'
            
        if stats_type not in self._STATS_CLASSES:
            available_types = list(self._STATS_CLASSES.keys())
            raise ValueError(f"Unknown stats_type '{stats_type}'. Available types: {available_types}")
            
        stats_class = self._STATS_CLASSES[stats_type]
            
        for key, value in kwargs.items():
            self._log[key] = value
            if key not in self._stats:
                self._lazy_init_stat(key, value, stats_class)
            self._stats[key].update(value)

    def get(self) -> Dict[str, Any]:
        """
        Return the current log dictionary. For keys tracked with running statistics, include '<key>_mean', '<key>_std'.
        """
        result = dict(self._log)
        for key, stat in self._stats.items():
            if stat.count > 0:
                result[f"{key}_mean"] = stat.mean
                result[f"{key}_std"] = stat.std
        return result

    def reset(self) -> None:
        """
        Reset the log for the next iteration. Running statistics are preserved until reset_stats is called.
        """
        self._log.clear()

    def reset_stats(self) -> None:
        """
        Reset running statistics.
        """
        self._stats.clear()

    def _lazy_init_stat(self, key: str, value: Any, stats_class: Type[RunningStatistics]) -> None:
        # Infer backend, shape, and dtype
        backend = None
        shape = None
        dtype = None
        try:
            import torch
            if isinstance(value, torch.Tensor):
                backend = "torch"
                shape = tuple(value.shape)
                dtype = value.dtype
        except ImportError:
            pass
        try:
            import numpy as np
            if backend is None and isinstance(value, np.ndarray):
                backend = "numpy"
                shape = tuple(value.shape)
                dtype = value.dtype
        except ImportError:
            pass
        if backend is None:
            if isinstance(value, (list, tuple)):
                backend = "python"
                shape = (len(value),)
                dtype = None
            elif isinstance(value, (float, int)):
                backend = "python"
                shape = ()
                dtype = None
            else:
                raise TypeError(f"Cannot infer backend for key '{key}' with value type {type(value)}")
        kwargs_stat = {"shape": shape, "backend": backend}
        if dtype is not None:
            kwargs_stat["dtype"] = dtype
        
        # Check if the stats_class supports additional parameters (like alpha for ExponentialMovingAverage)
        try:
            # For ExponentialMovingAverage, we might want to add alpha parameter in the future
            self._stats[key] = stats_class(**kwargs_stat)
        except TypeError as e:
            # If the stats_class doesn't accept the dtype parameter, try without it
            if "dtype" in str(e) and "dtype" in kwargs_stat:
                kwargs_stat.pop("dtype")
                self._stats[key] = stats_class(**kwargs_stat)
            else:
                raise e

    def __str__(self) -> str:
        return str(self.get())
        