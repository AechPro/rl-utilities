# RL Utilities

A collection of utilities for my reinforcement learning projects, providing logging, statistics tracking, neural network modules, and more.

## Features

### **Flexible Logging System**
- **RLDataLogger**: Core logger for tracking values and running statistics
- **Handlers**: Pluggable output handlers for different logging destinations (print, wandb, etc.). Allows for user-defined categories to format output.
- **CompositeLogger**: Allows composition of multiple handlers to e.g. both print to console and log to wandb.

### **Miscellaneous Utilities**
- **WelfordRunningStat**: Online statistics computation for data. Instantiate with a shape and backend ("torch", "numpy", or "python").
- **RollingTensor**: Maintains a rolling stack of tensors with a fixed shape. Useful for things like frame stacking in Atari.

### **Neural Network Components**
- **GaussianHead**: Gaussian head for learning gaussian distributions
- **QuantileHead**: Quantile head for quantile regression
- **FocalLoss**: Implementation of focal loss for imbalanced categorical datasets
- **QuantileLoss**: Loss function for quantile regression

### **Neural Network Factories**
- **build_ffnn_model**: Factory function for creating feedforward neural networks

### **Core Utilities**
- **AffineMap**: Fixed affine transformation module, useful for transforming variances in gaussian distributions to a positive interval.

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
import torch
import numpy as np

# For PyTorch tensors
stat_torch = WelfordRunningStat(shape=(3,), backend="torch")
stat_torch.update(torch.tensor([1.0, 2.0, 3.0]))
stat_torch.update(torch.tensor([4.0, 5.0, 6.0]))
print(f"Torch Mean: {stat_torch.mean}, Std: {stat_torch.std}")

# For numpy arrays
stat_np = WelfordRunningStat(shape=(3,), backend="numpy")
stat_np.update(np.array([1.0, 2.0, 3.0]))
stat_np.update(np.array([4.0, 5.0, 6.0]))
print(f"Numpy Mean: {stat_np.mean}, Std: {stat_np.std}")

# For python lists
stat_python = WelfordRunningStat(shape=(3,), backend="python")
stat_python.update([1.0, 2.0, 3.0])
stat_python.update([4.0, 5.0, 6.0])
print(f"Python Mean: {stat_python.mean}, Std: {stat_python.std}")

# For python scalars
stat_scalar = WelfordRunningStat(shape=(), backend="python")
stat_scalar.update(1.0)
stat_scalar.update(2.0)
print(f"Scalar Mean: {stat_scalar.mean}, Std: {stat_scalar.std}")
```

## Requirements

- Python >= 3.7
- PyTorch >= 1.7.0

### Optional Dependencies
- numpy >= 1.19.0 (for enhanced WelfordRunningStat support)
- wandb >= 0.12.0 (for WandbLoggerHandler)
