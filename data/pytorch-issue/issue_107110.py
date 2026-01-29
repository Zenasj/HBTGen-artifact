# torch.rand(2, 7, dtype=torch.int32).cuda()
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, src):
        # Replicate the destination tensor and its as_strided view
        dst = torch.full((7, 2), -1, device=src.device, dtype=src.dtype)
        dst_view = dst.as_strided((7, 2), (1, 2))  # stride (1,2) creates overlapping elements
        src_t = src.t()  # transpose to 7x2
        dst_view.copy_(src_t)  # perform copy with potential data race
        return dst

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random input matching the original example's src tensor
    return torch.randint(0, 15, (2, 7), dtype=torch.int32).cuda()

