# torch.rand(B, C, H, W, dtype=...)  # (4, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=1)
        self.ln = LayerNorm2d(96)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, groups=3)

    def forward(self, x):
        x = self.ln(self.conv1(x))
        o1 = self.conv2(x)
        o2 = self.conv2(x.contiguous())
        diff = o1 - o2
        return diff

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)

def my_model_function():
    return MyModel()

def GetInput():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.ones(4, 3, 224, 224, device=device, requires_grad=True)

