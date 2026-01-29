# torch.rand(8, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(224)  # Normalize over last dimension (size 224)

    def forward(self, x):
        return self.norm(x)

def my_model_function():
    # Create model and move to MPS if available
    model = MyModel()
    model.to("mps" if torch.backends.mps.is_available() else "cpu")
    return model

def GetInput():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.rand(8, 3, 224, 224, device=device, dtype=torch.float32)

