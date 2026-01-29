# torch.rand(5, dtype=torch.float32)  # Input shape is (5,)
import torch
import numpy as np

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Flags as in the original issue's example
        self.flag1 = torch.tensor([0., 1., 0., 1., 0.], dtype=torch.float32)  # Original torch.Tensor (float)
        self.flag2 = np.array([0, 1, 0, 1, 0], dtype=np.int32)  # Original numpy array

    def forward(self, x):
        # Compute boolean masks
        mask1 = self.flag1 == 1.0  # torch.BoolTensor
        mask2 = self.flag2 == 1    # numpy.bool_ array

        # Apply both indexing methods
        selected_torch = x[mask1]
        selected_numpy = x[mask2]  # Behaves differently in PyTorch <1.2

        # Compare results (shape and element-wise equality)
        same_shape = selected_torch.shape == selected_numpy.shape
        if not same_shape:
            return torch.tensor(False)  # Shape mismatch indicates difference
        return torch.all(selected_torch == selected_numpy)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, dtype=torch.float32)  # Matches input shape (5,)

