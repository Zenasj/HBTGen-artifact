# torch.rand(B, C, H, dtype=torch.float32)
import torch
import torch.nn as nn

class RepeatInterleaveModel(nn.Module):
    def forward(self, x):
        return x.repeat_interleave(2, dim=-1)

class UnsqueezeRepeat(nn.Module):
    def forward(self, x):
        repeats = tuple((1,) * len(x.shape) + (2,))
        return x.unsqueeze(-1).repeat(repeats).flatten(-2, -1)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = RepeatInterleaveModel()
        self.model2 = UnsqueezeRepeat()
    
    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        # Return boolean tensor indicating equality
        return torch.tensor(torch.equal(out1, out2), dtype=torch.bool).unsqueeze(0)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 4, 32, dtype=torch.float32)

