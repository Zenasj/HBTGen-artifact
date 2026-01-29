# torch.rand(3, 64, 64, dtype=torch.float32)

import torch
from torch import nn
from typing import Sequence

@torch.library.custom_op("mylib::crop", mutates_args=())
def crop(pic: torch.Tensor, box: Sequence[int]) -> torch.Tensor:
    channels = pic.shape[0]
    x0, y0, x1, y1 = box
    result = pic[:, y0:y1, x0:x1].permute(1, 2, 0).contiguous().permute(2, 0, 1)
    return result

@crop.register_fake
def _(pic, box):
    channels = pic.shape[0]
    x0, y0, x1, y1 = box
    result = pic.new_empty(y1 - y0, x1 - x0, channels).permute(2, 0, 1)
    return result

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.box = (10, 10, 50, 50)  # Fixed crop coordinates from the issue

    def forward(self, x):
        return crop(x, self.box)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 64, 64, dtype=torch.float32)

