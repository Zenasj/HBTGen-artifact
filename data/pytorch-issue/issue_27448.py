# torch.rand(B, 512, 4, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv(x)
        return torch.rsqrt(x)  # Simulates problematic rsqrt operation in StyleGAN

def my_model_function():
    model = MyModel()
    # Initialize weights to avoid NaN in rsqrt
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                param.normal_(mean=0.1, std=0.02)  # StyleGAN-like initialization
    return model

def GetInput():
    return torch.rand(1, 512, 4, 4, dtype=torch.float32)

