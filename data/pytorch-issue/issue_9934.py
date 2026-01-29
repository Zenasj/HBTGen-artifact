# torch.rand(2, 2, dtype=torch.float32, device='cuda')
import torch
import numpy as np

class MyModel(torch.nn.Module):
    def forward(self, x):
        # Compare implicit vs explicit NumPy conversion behaviors
        try:
            # Implicit conversion via __array__ (may auto-CPU)
            implicit_arr = np.linalg.inv(x)
            implicit_tensor = torch.from_numpy(implicit_arr)
        except:
            implicit_tensor = None

        # Explicit conversion (manual CPU move)
        explicit_arr = x.cpu().numpy()
        explicit_arr_inv = np.linalg.inv(explicit_arr)
        explicit_tensor = torch.from_numpy(explicit_arr_inv)

        # Return comparison result
        if implicit_tensor is not None:
            return torch.allclose(implicit_tensor, explicit_tensor)
        else:
            return torch.tensor(False, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2, dtype=torch.float32, device='cuda')

