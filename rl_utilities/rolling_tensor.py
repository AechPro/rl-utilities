import torch
from typing import Tuple, Union, Optional

class RollingTensor(object):
    def __init__(self, shape: Tuple[int, ...], stack_num: int = 1, device: Union[str, torch.device] = "cpu") -> None:
        self.shape = shape
        self.stack_num = stack_num
        self.device = device
        self.tensor: Optional[torch.Tensor] = None
        self.reset()

    def stack(self, tensor_to_stack: torch.Tensor) -> None:            
        if self.tensor is None:
            self.reset(tensor_to_stack)
        elif self.stack_num == 1:
            self.tensor = tensor_to_stack.unsqueeze(0).unsqueeze(0)
        else:
            self.tensor = torch.roll(self.tensor, shifts=-1, dims=1)
            self.tensor[:, -1, ...] = tensor_to_stack

    def reset(self, initial_tensor: Optional[torch.Tensor] = None) -> None:
        if initial_tensor is None:
            self.tensor = torch.zeros((1, self.stack_num, *self.shape), 
                                       dtype=torch.float32, device=self.device)
        else:
            initial_tensor = torch.as_tensor(initial_tensor, dtype=torch.float32, device=self.device)
            
            if initial_tensor.shape[1:] == (self.stack_num, *self.shape):
                self.tensor = initial_tensor.clone()
            else:
                if initial_tensor.dim() == len(self.shape):
                    initial_tensor = initial_tensor.unsqueeze(0)
                    
                if self.stack_num == 1:
                    self.tensor = initial_tensor
                else:
                    self.tensor = torch.zeros((initial_tensor.shape[0], self.stack_num, *self.shape), 
                                                dtype=torch.float32, device=self.device)
                    self.tensor[:, -1, ...] = initial_tensor
