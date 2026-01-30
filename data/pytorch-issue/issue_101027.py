import torch
import torch.nn as nn

class Foo(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.linear = nn.Linear(15, 30)
    
    def forward(self, x: torch.Tensor):
        x *= x
        y = self.linear(x)
        y += 3
        y -= 1
        z = torch.mean(y)
        return z