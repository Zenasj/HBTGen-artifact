# torch.rand(100, dtype=torch.float32).cuda()
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Compute CPU and GPU outputs
        cpu_out = torch.cumprod(x.cpu(), dim=0)
        gpu_out = torch.cumprod(x, dim=0).cpu()
        
        # Compare element-wise, treating NaNs as non-equal
        equal = torch.all(torch.eq(cpu_out, gpu_out))
        return torch.tensor(not equal, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Replicate the input from the issue (arange(0,50,0.5) on CUDA)
    return torch.arange(0, 50, 0.5, dtype=torch.float32).cuda()

