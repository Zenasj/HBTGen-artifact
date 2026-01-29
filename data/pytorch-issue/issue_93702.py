# torch.rand(1, 2, 16, 16, dtype=torch.float32, device='cuda', requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1, inplace=False)

    def forward(self, inputs: torch.Tensor):
        mask1 = torch.ones((1, 16), device='cuda')
        mask2 = torch.full((1, 16), 3.0, device='cuda')
        out1 = inputs + mask1
        out2 = torch.nn.functional.softmax(out1, dim=-1)
        out3 = out2 - mask2
        out4 = self.dropout(out3)
        return out4

def my_model_function():
    return MyModel().cuda()

def GetInput():
    return torch.randn(1, 2, 16, 16, device='cuda', requires_grad=True)

