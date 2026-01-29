# torch.rand(B=4, C=2, H=40, W=40, dtype=torch.float32) ← Input shape inferred from issue's sample code
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Apply unfold followed by fold to test both im2col and col2im operations
        unfolded = F.unfold(x, kernel_size=3, padding=1, stride=1)
        folded = F.fold(unfolded, output_size=(40, 40), kernel_size=3, padding=1, stride=1)
        return folded

def my_model_function():
    # Returns a model instance that combines unfold/fold operations
    return MyModel()

def GetInput():
    # Returns MPS/CPU tensor matching the model's input requirements
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.rand(4, 2, 40, 40, dtype=torch.float32, device=device)

