# torch.rand((), dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compare CPU vs CUDA conversion of NaN to int32
        cpu_val = x.cpu().to(torch.int32)
        cuda_val = x.cuda().to(torch.int32).cpu()  # Bring back to CPU for comparison
        return (cpu_val != cuda_val).to(torch.int32)  # Return 1 if different, 0 otherwise

def my_model_function():
    return MyModel()

def GetInput():
    # Create a NaN tensor as input
    return torch.tensor(float('nan'))

