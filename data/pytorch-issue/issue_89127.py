# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.device_mps = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        self.device_cpu = torch.device('cpu')

    def forward(self, x):
        # Generate tensor directly on MPS (problematic path)
        mps_tensor = torch.normal(0, 1, size=(1000, 1000), device=self.device_mps)
        # Generate tensor on CPU then move to MPS (non-problematic path)
        cpu_moved = torch.normal(0, 1, size=(1000, 1000), device=self.device_cpu).to(self.device_mps)
        
        # Check if MPS path produces NaNs (core bug comparison)
        has_nan_mps = torch.any(torch.isnan(mps_tensor))
        # Return boolean tensor indicating presence of NaNs in MPS path
        return torch.tensor([has_nan_mps], dtype=torch.bool, device=self.device_mps)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, device=torch.device('cpu'))  # Dummy input tensor

