import torch
import torch.nn as nn
from typing import Tuple
from rl_utilities import AffineMap

class GaussianHead(nn.Module):
    def __init__(self, output_dim: int, stdev_range: Tuple[float, float] = (0.1, 1)) -> None:
        """
        A torch module that splits an input into a mean and a standard deviation. The standard deviation is mapped to the range `stdev_range` using an affine map.

        :param output_dim: Number of dimensions in the output space. It is expected that the shape of any input to this module will be (dim0, dim1, ... , output_dim*2).
        """
        super().__init__()
        self.stdev_map = AffineMap(from_range=(-1, 1), to_range=stdev_range) # Tanh range is (-1, 1).
        self.tanh = nn.Tanh()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: Input tensor of shape (dim0, dim1, ... , output_dim*2). Note that the second half of the input will be passed through a tanh function 
                  before being mapped to the range `stdev_range`.

        :return: Tuple of mean and standard deviation tensors, both with shape (dim0, dim1, ... , output_dim).
        """

        mean = x[..., :self.output_dim]
        stdev = self.stdev_map(self.tanh(x[..., self.output_dim:]))
        return mean, stdev
