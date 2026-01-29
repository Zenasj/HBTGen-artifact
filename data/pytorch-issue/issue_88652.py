import torch
import numpy as np

# torch.rand(3, dtype=torch.int64)
class MyModel(torch.nn.Module):
    def forward(self, input_params):
        start = input_params[0].item()
        end = input_params[1].item()
        num = input_params[2].item()
        # Compute PyTorch's linspace (CPU-specific issue)
        torch_result = torch.linspace(start, end, num, dtype=torch.int64)
        # Compute NumPy's linspace (reference)
        np_result = np.linspace(start, end, num, dtype=np.int64)
        np_tensor = torch.from_numpy(np_result)
        # Return boolean indicating exact match
        return torch.all(torch.eq(torch_result, np_tensor))

def my_model_function():
    return MyModel()

def GetInput():
    # Parameters from original issue (start=1, end=32, num=10)
    return torch.tensor([1, 32, 10], dtype=torch.int64)

