# torch.randn(1, 1, 256, 256, 4, dtype=torch.float32, device='cuda')  # Fails with CUDA error for Z >=4
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Conv3d(1, 64, 5, padding=2)
        self.layer2 = nn.Conv3d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_vol):
        first = self.layer1(in_vol)
        MonoClass = self.layer2(first)
        Mask = self.sigmoid(MonoClass)
        return Mask

def my_model_function():
    # Returns model initialized on CUDA (as in original issue's failing case)
    return MyModel().cuda()

def GetInput():
    # Returns a CUDA tensor matching the failing input dimensions
    return torch.randn(1, 1, 256, 256, 4, dtype=torch.float32, device='cuda')

