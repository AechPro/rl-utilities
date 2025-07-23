import torch.nn as nn
import torch

class QuantileLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        :param pred: Tensor containing predicted quantile values. Must have shape (batch_size, ..., n_quantiles, output_dim).
        :param target: Tensor containing target values. Will be reshaped to (batch_size, ..., 1, output_dim).
        :param tau: Tensor containing quantiles to query the model with. Must have shape (batch_size, ..., n_quantiles, 1) in [0, 1].
        """
        expected_target_shape = list(pred.shape[:-2]) + [1, pred.shape[-1]]
        target = target.view(*expected_target_shape)
        
        diff = target - pred  # [batch, ..., n_quantiles, output_dim]
        loss = torch.where(diff > 0, tau * diff, (1 - tau) * -diff)
        return loss.mean()