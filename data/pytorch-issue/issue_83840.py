# torch.rand(1, 3, 64, 64, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, x):
        out_true = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)
        out_false = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False, antialias=False)
        return torch.tensor(torch.allclose(out_true, out_false, atol=1e-5), dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randint(0, 256, (3, 64, 64), dtype=torch.float32)
    x = x.as_strided(x.size(), stride=(1, 192, 3))
    x = x[None, :, :, :]
    return x

