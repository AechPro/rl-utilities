import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

class QuantileHead(nn.Module):
    """
    Generic quantile regression head for distributional RL (IQN-style).
    If input_dim==0, uses quantile embedding alone as input.
    If input_dim>0, uses IQN-style mixing of features and quantile embedding.
    """
    def __init__(self, input_dim: int, output_dim: int, n_quantiles: int = 32, n_basis: int = 64, hidden_dim: int = 128, device: Union[str, torch.device] = 'cpu') -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_quantiles = n_quantiles
        self.n_basis = n_basis
        self.hidden_dim = hidden_dim
        

        # Quantile embedding (cosine basis)
        self.phi = nn.Sequential(
            nn.Linear(n_basis, hidden_dim),
            nn.ReLU()
        ).to(device)

        if input_dim > 0:
            self.feature_fc = nn.Linear(input_dim, hidden_dim).to(device)

        self.out_fc = nn.Linear(hidden_dim, output_dim).to(device)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, tau: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch_size, ..., input_dim).
        :param tau: Tensor containing quantiles to query the model with. Must have shape (batch_size, ..., n_quantiles, 1) in [0, 1]. 
                    If tau is not provided, quantiles will be sampled uniformly on [0, 1].

        :return: Tensor containing predicted quantile values of shape (batch_size, ..., output_dim).
        """
        if tau is None:
            tau = torch.rand(x.shape[0], self.n_quantiles, 1).to(x.device)

        n_quantiles = tau.shape[-2]
        cos_basis = torch.arange(1, self.n_basis + 1, device=tau.device).float()
        quantile_embed = torch.cos(tau * cos_basis * torch.pi)  # [batch, ..., n_quantiles, n_basis]
        quantile_embed = self.phi(quantile_embed)  # [batch, ..., n_quantiles, D]
        if self.input_dim > 0:
            expand_shape = list(x.shape[:-1]) + [n_quantiles, x.shape[-1]]
            x = x.unsqueeze(-2).expand(*expand_shape)
            x = self.feature_fc(x)
            x = x * quantile_embed
        else:
            x = quantile_embed
        x = self.activation(x)
        quantile_values = self.out_fc(x)
        return quantile_values


    

    def sample_from_quantiles(self, quantiles: torch.Tensor) -> torch.Tensor:
        """Randomly sample from the quantile outputs."""
        batch_size = quantiles.size(0)
        random_indices = torch.randint(0, self.n_quantiles, (batch_size,), device=quantiles.device)
        sampled_values = quantiles[range(batch_size), random_indices, :]
        return sampled_values
