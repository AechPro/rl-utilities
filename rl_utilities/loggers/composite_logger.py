from typing import List, Dict, Optional, Any
from .rl_data_logger import RLDataLogger

class LoggerHandler:
    """Abstract base class for logger handlers."""
    def handle(self, logger: RLDataLogger, **kwargs: Any) -> None:
        raise NotImplementedError

class PrintLoggerHandler(LoggerHandler):
    def __init__(self, categories: Optional[Dict[str, List[str]]] = None) -> None:
        self.categories = categories or {}

    def handle(self, logger: RLDataLogger, **kwargs: Any) -> None:
        data = logger.get()
        printed_keys = set()
        if self.categories:
            for cat, keys in self.categories.items():
                cat_items = [(k, data[k]) for k in keys if k in data]
                if cat_items:
                    print(f"=== {cat} ===")
                    for k, v in cat_items:
                        print(f"{k:20}: {v}")
                    print()
                    printed_keys.update(keys)
        uncategorized = [(k, v) for k, v in data.items() if k not in printed_keys]
        if uncategorized:
            print("=== Uncategorized ===")
            for k, v in uncategorized:
                print(f"{k:20}: {v}")
            print()

class WandbLoggerHandler(LoggerHandler):
    def __init__(self, categories: Optional[Dict[str, List[str]]] = None, wandb_run: Optional[Any] = None, **wandb_kwargs: Any) -> None:
        import wandb
        self.categories = categories or {}
        self.run = wandb_run if wandb_run is not None else wandb.init(**wandb_kwargs)

    def handle(self, logger: RLDataLogger, step: Optional[int] = None, **kwargs: Any) -> None:
        data = logger.get()
        log_dict = {}
        used_keys = set()
        # Add categorized keys as 'category/key'
        if self.categories:
            for cat, keys in self.categories.items():
                for k in keys:
                    if k in data:
                        log_dict[f"{cat}/{k}"] = data[k]
                        used_keys.add(k)
        # Add uncategorized keys as normal
        for k, v in data.items():
            if k not in used_keys:
                log_dict[k] = v
        if step is not None:
            self.run.log(log_dict, step=step)
        else:
            self.run.log(log_dict)

class CompositeLogger:
    def __init__(self, base_logger: RLDataLogger, handlers: List[LoggerHandler]) -> None:
        self.base_logger = base_logger
        self.handlers = handlers

    def log(self, **kwargs: Any) -> None:
        self.base_logger.log(**kwargs)

    def log_stats(self, **kwargs: Any) -> None:
        self.base_logger.log_stats(**kwargs)

    def reset(self) -> None:
        self.base_logger.reset()

    def reset_stats(self) -> None:
        self.base_logger.reset_stats()

    def get(self) -> Dict[str, Any]:
        return self.base_logger.get()

    def handle_all(self, **kwargs: Any) -> None:
        for handler in self.handlers:
            handler.handle(self.base_logger, **kwargs)
