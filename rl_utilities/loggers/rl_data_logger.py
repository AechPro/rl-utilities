from collections import defaultdict
from typing import Dict, Any
from rl_utilities import WelfordRunningStat

class RLDataLogger:
    def __init__(self) -> None:
        """
        RLDataLogger for RL projects. Tracks per-key values and optional running statistics.
        """
        self._log: Dict[str, Any] = {}
        self._stats: Dict[str, WelfordRunningStat] = defaultdict(WelfordRunningStat)

    def log(self, **kwargs: Any) -> None:
        """
        Log key-value pairs. Accepts multiple named arguments.
        Example: logger.log(loss=0.5, reward=1.0)
        """
        for key, value in kwargs.items():
            self._log[key] = value

    def log_stats(self, **kwargs: Any) -> None:
        """
        Log key-value pairs and update their running statistics (mean, std).
        Example: logger.log_stats(loss=0.5, reward=1.0)
        Supports numbers, lists, numpy arrays, torch tensors, etc.
        """
        for key, value in kwargs.items():
            self._log[key] = value
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

    def __str__(self) -> str:
        return str(self.get())
        