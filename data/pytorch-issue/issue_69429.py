# torch.rand(320, 320, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 320
        self.a = nn.Conv2d(channels, channels, 5, padding=2, bias=False, groups=channels)
        self.a.eval()  # Matches original code's .eval() state

    def forward(self, x):
        a_out = self.a(x)
        y1 = torch.maximum(x, a_out)
        y2 = x + F.relu(a_out - x)
        y3 = a_out + F.relu(x - a_out)
        diff_y1y2 = torch.sum(y1 - y2)
        diff_y1y3 = torch.sum(y1 - y3)
        return diff_y1y2, diff_y1y3  # Returns comparison differences as per original code

def my_model_function():
    return MyModel()

def GetInput():
    channels = 320
    hw = 1
    return torch.rand(channels, channels, hw, hw, dtype=torch.float32, requires_grad=False)

