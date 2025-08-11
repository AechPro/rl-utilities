import torch 
import torch.nn as nn 

class GLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, activation: nn.Module = nn.SiLU()) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.activation(self.fc1(x)) * self.fc2(x))