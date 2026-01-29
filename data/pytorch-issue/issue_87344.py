# torch.rand(3, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward_train(self, x, gt_label, **kwargs):
        y = self.linear(x + self.param).clamp(min=0.0, max=1.0)
        diff = y - gt_label
        return torch.mean(diff)

    def forward(self, x, return_loss=False, **kwargs):
        if return_loss:
            return self.forward_train(x, **kwargs)
        else:
            return self.linear(x + self.param).clamp(min=0.1, max=1.0)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 4, dtype=torch.float32)

