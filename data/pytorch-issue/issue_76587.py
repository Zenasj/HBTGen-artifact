# torch.rand(B, C, L, dtype=torch.float32)  # Input shape is 3D (batch, channels, length)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1)
    
    def forward(self, x):
        # Compare outputs between normal and autograd inputs
        # Return 1.0 if outputs match, 0.0 if error occurs or outputs differ
        try:
            # Non-grad path
            t1 = x.clone().detach()
            out1 = self.pool(t1)
            
            # Grad path (may raise error)
            t2 = x.clone().detach().requires_grad_()
            out2 = self.pool(t2)
            
            return torch.tensor(1.0) if torch.allclose(out1, out2) else torch.tensor(0.0)
        except Exception:
            return torch.tensor(0.0)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape from the issue (17, 0, 50)
    return torch.rand(17, 0, 50, dtype=torch.float32)

