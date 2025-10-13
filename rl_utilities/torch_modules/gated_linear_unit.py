import torch 
import torch.nn as nn 

class GLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, activation: nn.Module = nn.Sigmoid(), use_layer_norm=True) -> None:
        super().__init__()
        self.layer_norm = None
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(normalized_shape=(hidden_dim,))
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preactivation = self.fc1(x)
        if self.layer_norm is not None:
            preactivation = self.layer_norm(preactivation)
        return self.activation(preactivation) * self.fc2(x)