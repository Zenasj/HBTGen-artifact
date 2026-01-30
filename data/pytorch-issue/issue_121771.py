import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.linear = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear.forward(x) # Error
        # return self.linear(x) # OK