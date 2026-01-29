# torch.rand(9, 8, dtype=torch.half)
import torch
import numpy as np

class MyModel(torch.nn.Module):
    def forward(self, x):
        # Compute PyTorch sum
        pt_sum = torch.sum(x, dtype=torch.half)
        
        # Compute NumPy sum (convert to NumPy array and back to tensor)
        np_array = x.detach().cpu().numpy().astype(np.float16)
        np_sum = np.nansum(np_array)
        np_sum_tensor = torch.tensor(np_sum, dtype=torch.half)
        
        # Compare and return True if difference exceeds tolerance (1.0)
        return torch.abs(pt_sum - np_sum_tensor) > 1.0

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((9, 8), dtype=torch.half)

