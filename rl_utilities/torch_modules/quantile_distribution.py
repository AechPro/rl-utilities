from torch.distributions import Distribution
import torch
from typing import Tuple

class QuantileDistribution(Distribution):
    def __init__(self, quantile_values: torch.Tensor, tau: torch.Tensor) -> None:
        super().__init__()
        self.quantile_values = quantile_values  
        self.tau = tau  

        self.left = quantile_values[..., :-1, :]  
        self.right = quantile_values[..., 1:, :]  
        self.tau_left = tau[..., :-1, :]          
        self.tau_right = tau[..., 1:, :]          
        self.probs = (self.tau_right - self.tau_left).clamp(min=1e-8)  
        self.widths = (self.right - self.left).clamp(min=1e-8)         
        self.densities = self.probs / self.widths                      

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value = value.unsqueeze(-2)
        interval_mask = (value >= self.left) & (value < self.right)

        last_bin = (value == self.right) & (self.right == self.right.max(dim=-2, keepdim=True).values)
        interval_mask = interval_mask | last_bin

        density = (self.densities * interval_mask.float()).sum(dim=-2)
        log_prob = torch.log(density + 1e-12)

        return log_prob

    def sample(self, sample_shape: torch.Size = torch.Size((1, ))) -> torch.Tensor:
        n_quantiles = self.quantile_values.shape[-2]
        flat_shape = self.quantile_values.shape[:-2] + sample_shape

        idx = torch.randint(0, n_quantiles, flat_shape, device=self.quantile_values.device)
        idx_expanded = idx.unsqueeze(-1).expand(*flat_shape, self.quantile_values.shape[-1])

        sample = torch.gather(self.quantile_values, -2, idx_expanded)

        return sample

    def entropy(self) -> torch.Tensor:
        entropy = -(self.probs * torch.log(self.densities + 1e-12)).sum(dim=-2)
        return entropy

    @property
    def batch_shape(self) -> torch.Size:
        return self.quantile_values.shape[:-1]

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.quantile_values.shape[-1],)

    @property
    def mean(self) -> torch.Tensor:
        return self.quantile_values.mean(dim=-2)

    @property
    def stddev(self) -> torch.Tensor:
        bin_means = 0.5 * (self.left + self.right)
        bin_vars = (self.widths ** 2) / 12 # divide by 12 because the variance of a uniform distribution is 12, and the quantile bins are uniformly distributed.
        mean = self.mean.unsqueeze(-2)
        
        weighted_var = (self.probs * (bin_vars + (bin_means - mean) ** 2)).sum(dim=-2)
        stddev = torch.sqrt(weighted_var)
        return stddev

    @property
    def support(self) -> torch.Tensor:
        min_val = self.quantile_values.min(dim=-2).values
        max_val = self.quantile_values.max(dim=-2).values
        return torch.stack([min_val, max_val], dim=0)