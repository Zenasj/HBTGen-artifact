# torch.rand(1024, 1, 1024, 1024, dtype=torch.half, device='cuda')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.ConvTranspose2d(1, 1, 1, 1, bias=False)
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = MyModel()
    model = model.cuda().half()  # Move to CUDA and use FP16
    return model

def GetInput():
    return torch.randn(1024, 1, 1024, 1024, dtype=torch.half, device='cuda')

