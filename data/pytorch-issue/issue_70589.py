# torch.rand(64, 3, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, 3, 3, 3))  # Replicate the example's weight initialization

    def forward(self, x):
        # CPU computation
        x_cpu = x.to('cpu')
        w_cpu = self.weight.to('cpu')
        out_cpu = F.conv2d(x_cpu, w_cpu, stride=1, padding=1)
        
        # GPU computation (if available)
        if torch.cuda.is_available():
            x_gpu = x.to('cuda')
            w_gpu = self.weight.to('cuda')
            out_gpu = F.conv2d(x_gpu, w_gpu, stride=1, padding=1).cpu()
            max_diff = torch.max(torch.abs(out_cpu - out_gpu))
            return max_diff
        else:
            return torch.tensor(0.0)  # Return zero difference if no GPU

def my_model_function():
    # Seed setup to replicate the example's initial conditions
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    return MyModel()

def GetInput():
    # Replicate the example's input initialization
    torch.manual_seed(0)
    return torch.randn(64, 3, 32, 32)

