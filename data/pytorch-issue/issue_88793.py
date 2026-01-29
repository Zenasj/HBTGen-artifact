# torch.rand(3, dtype=torch.int32)  # Input is a 1D tensor of integers
import numpy as np
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        results = []
        # Test CharStorage case
        try:
            torch.save(x, torch.CharStorage())
        except Exception:
            results.append(1)  # 1 indicates exception occurred
        else:
            results.append(0)
        
        # Test numpy array case
        try:
            arr = x.numpy()
            torch.save(arr, arr)
        except Exception:
            results.append(1)
        else:
            results.append(0)
        
        return torch.tensor(results, dtype=torch.int32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 100, (3,), dtype=torch.int32)

