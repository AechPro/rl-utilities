import torch
import torch.nn as nn
from typing import Tuple, Optional

class AffineMap(nn.Module):
    """
    Utility torch module for affine mapping from one range to another.
    """
    def __init__(self, from_range: Tuple[float, float] = (-1, 1), to_range: Tuple[float, float] = (0.1, 1)) -> None:
        super().__init__()
        self.from_range = from_range
        self.to_range = to_range

    def forward(self, x: torch.Tensor, from_range: Optional[Tuple[float, float]] = None, to_range: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        if from_range is None:
            from_range = self.from_range
        if to_range is None:
            to_range = self.to_range
            
        return (x - from_range[0]) * (to_range[1] - to_range[0]) / (from_range[1] - from_range[0]) + to_range[0]
