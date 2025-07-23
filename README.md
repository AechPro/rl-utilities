# RL Utilities

A flexible collection of utilities for reinforcement learning projects, providing logging, statistics tracking, neural network modules, and more.

## Features

### 🎯 **Flexible Logging System**
- **RLDataLogger**: Core logger for tracking values and running statistics
- **CompositeLogger**: Multi-sink logging (print, wandb, etc.) with a single interface
- **Handlers**: Pluggable output handlers for different destinations
- **Categories**: Organize logged data into user-defined categories

### 📊 **Statistical Utilities**
- **WelfordRunningStat**: Robust online statistics computation supporting various data types (tensors, arrays, lists)
- **RollingTensor**: Efficient rolling window operations for PyTorch tensors

### 🧠 **Neural Network Components**
- **GaussianHead**: Neural network head for Gaussian distributions
- **QuantileHead**: Quantile-based distribution heads
- **FocalLoss**: Implementation of focal loss for imbalanced datasets
- **QuantileLoss**: Loss functions for quantile regression

### 🏭 **Neural Network Factories**
- **build_ffnn_model**: Factory function for creating feedforward neural networks

### 🔧 **Core Utilities**
- **AffineMap**: Learnable affine transformations

## Installation

```bash
pip install rl-utilities
```

For full functionality including wandb integration:

```bash
pip install rl-utilities[full]
```

## Quick Start

### Basic Logging

```python
from rl_utilities.loggers import RLDataLogger

logger = RLDataLogger()

# Log values
logger.log(loss=0.1, reward=1.0)

# Track running statistics
logger.log_stats(loss=0.1)
logger.log_stats(loss=0.08)

# Get current data
data = logger.get()
print(data)  # {'loss': 0.08, 'reward': 1.0, 'loss_mean': 0.09, 'loss_std': 0.01}
```

### Multi-Sink Logging

```python
from rl_utilities.loggers import RLDataLogger, CompositeLogger
from rl_utilities.loggers.composite_logger import PrintLoggerHandler, WandbLoggerHandler
import wandb

# Create base logger and handlers
base_logger = RLDataLogger()
handlers = [
    PrintLoggerHandler(categories={"Losses": ["loss"], "Metrics": ["reward"]}),
    WandbLoggerHandler(categories={"Training": ["loss", "reward"]}, project="my-project")
]

# Create composite logger
logger = CompositeLogger(base_logger, handlers)

# Log once, output to multiple destinations
logger.log(loss=0.1, reward=1.0)
logger.log_stats(loss=0.1)

# Print formatted output and log to wandb
logger.handle_all()
```

### Neural Network Components

```python
import torch
from rl_utilities.torch_modules import GaussianHead, QuantileHead

# Gaussian distribution head
gaussian_head = GaussianHead(input_dim=64, output_dim=2)
mean, std = gaussian_head(torch.randn(32, 64))

# Quantile distribution head
quantile_head = QuantileHead(input_dim=64, output_dim=1, num_quantiles=51)
quantiles = quantile_head(torch.randn(32, 64))
```

### Running Statistics

```python
from rl_utilities import WelfordRunningStat

stat = WelfordRunningStat()

# Works with various data types
stat.update([1, 2, 3])
stat.update(torch.tensor([4, 5, 6]))

print(f"Mean: {stat.mean}, Std: {stat.std}")
```

## API Reference

### Logging

#### RLDataLogger
- `log(**kwargs)`: Log key-value pairs
- `log_stats(**kwargs)`: Log values and update running statistics
- `get()`: Get current logged data with statistics
- `reset()`: Clear current log
- `reset_stats()`: Clear running statistics

#### CompositeLogger
- `log(**kwargs)`: Log to base logger
- `log_stats(**kwargs)`: Log with statistics tracking
- `handle_all(**kwargs)`: Send data to all handlers

### Handlers

#### PrintLoggerHandler
- Formats and prints logged data by category
- `handle(logger, **kwargs)`: Process and print data

#### WandbLoggerHandler  
- Logs data to Weights & Biases
- Categories are logged as "category/key" for dashboard organization
- `handle(logger, step=None, **kwargs)`: Log to wandb

## Requirements

- Python >= 3.7
- PyTorch >= 1.7.0

### Optional Dependencies
- numpy >= 1.19.0 (for enhanced WelfordRunningStat support)
- wandb >= 0.12.0 (for WandbLoggerHandler)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
