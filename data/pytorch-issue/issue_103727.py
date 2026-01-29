# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class AxialModule(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.f_qr = nn.Parameter(torch.randn(in_channels), requires_grad=False)
        self.f_kr = nn.Parameter(torch.randn(in_channels), requires_grad=False)
        self.f_sve = nn.Parameter(torch.randn(in_channels), requires_grad=False)
        self.f_sv = nn.Parameter(torch.randn(in_channels), requires_grad=False)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.axial_height_1 = AxialModule()
        self.axial_width_1 = AxialModule()
        self.axial_height_2 = AxialModule()
        self.axial_width_2 = AxialModule()
    
    def forward(self, x):
        # Example computation using parameters from all modules
        combined = (
            self.axial_height_1.f_qr.view(1, -1, 1, 1) +
            self.axial_height_1.f_kr.view(1, -1, 1, 1) +
            self.axial_width_1.f_qr.view(1, -1, 1, 1) +
            self.axial_width_1.f_kr.view(1, -1, 1, 1) +
            self.axial_height_2.f_qr.view(1, -1, 1, 1) +
            self.axial_width_2.f_sv.view(1, -1, 1, 1)
        )
        return x + combined  # Minimal forward to involve parameters in computation

def my_model_function():
    return MyModel()

def GetInput():
    # Batch size 2, 3 channels, 224x224 image dimensions
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

