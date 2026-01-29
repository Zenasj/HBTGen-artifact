# torch.rand(1, 1, 3, 3, dtype=torch.float32)
import torch
import numpy as np

class MyModel(torch.nn.Module):
    def forward(self, x):
        # Compute unique on CPU and GPU
        cpu_unique = torch.unique(x.to('cpu'))
        gpu_unique = torch.unique(x.to('cuda')).to('cpu')  # Move GPU result to CPU for comparison
        
        # Sort both for element-wise comparison
        cpu_sorted, _ = torch.sort(cpu_unique)
        gpu_sorted, _ = torch.sort(gpu_unique)
        
        # Create mask where elements are equal or both NaN
        mask = (torch.isnan(cpu_sorted) & torch.isnan(gpu_sorted)) | (cpu_sorted == gpu_sorted)
        return torch.all(mask)

def my_model_function():
    return MyModel()

def GetInput():
    # Create a tensor with inf, nan, and 5 as in the original example
    data = np.array([[[[float('inf'), float('nan'), 5],
                      [float('inf'), float('nan'), 5],
                      [float('inf'), float('nan'), 5]]]], dtype=np.float32)
    return torch.tensor(data, dtype=torch.float32)

