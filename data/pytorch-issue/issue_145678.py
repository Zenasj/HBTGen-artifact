# torch.rand(1, 10, dtype=torch.float32)
import torch
import numpy as np

class MyModel(torch.nn.Module):
    def forward(self, x):
        # Compare CPU and GPU execution outcomes (success/failure)
        cpu_ok = True
        try:
            _ = x.to('cpu')[range(x.shape[1]), 0]
        except:
            cpu_ok = False

        gpu_ok = True
        try:
            _ = x.to('cuda:0')[range(x.shape[1]), 0]
        except:
            gpu_ok = False

        # Return True if both devices had the same outcome (both failed/succeeded)
        return torch.tensor(cpu_ok == gpu_ok, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 10, dtype=torch.float32)

