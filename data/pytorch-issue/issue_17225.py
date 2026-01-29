# torch.rand(5, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encapsulate both problematic (sigmoid) and working (tanh) models as submodules
        self.problematic = nn.Sequential()  # Empty container for sigmoid (intrinsic to PyTorch)
        self.working = nn.Tanh()  # Working alternative activation
        
    def forward(self, x):
        # Run both activations and return their outputs for comparison
        out_problematic = torch.sigmoid(x)  # Problematic path causing hang in multiprocessing
        out_working = self.working(x)       # Working path (tanh)
        return out_problematic, out_working

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape used in the original issue's reproduction script
    return torch.randn(5, 3, dtype=torch.float32)

