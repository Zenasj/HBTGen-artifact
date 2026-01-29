# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class IndexSelectModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('channel_order', torch.tensor([2, 1, 0]))

    def forward(self, x):
        return torch.index_select(x, 1, self.channel_order)

class MaxPoolModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('channel_order', torch.tensor([2, 1, 0]))
        self.avgpool = nn.AdaptiveMaxPool2d((4, 4))

    def forward(self, x):
        x1 = torch.index_select(x, 1, self.channel_order)
        return self.avgpool(x1)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.index_select = IndexSelectModule()
        self.max_pool = MaxPoolModule()

    def forward(self, x):
        # Return outputs of both modules to encapsulate comparison context
        return self.index_select(x), self.max_pool(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((2, 3, 16, 16), dtype=torch.float32)

