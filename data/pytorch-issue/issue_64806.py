# torch.rand(1, 512, 28, 28, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        pool_scales = (1, 2, 3, 6)  # Scales from the issue's code snippet
        self.ppm_pooling = nn.ModuleList([
            nn.AdaptiveAvgPool2d(scale) for scale in pool_scales
        ])

    def forward(self, x):
        outputs = []
        for pool in self.ppm_pooling:
            outputs.append(pool(x))
        return outputs  # Returns list of pooled tensors for each scale

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 512, 28, 28, dtype=torch.float32)  # Matches input shape assumptions

