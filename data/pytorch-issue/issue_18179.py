import torch
import torch.nn as nn

# torch.rand(B, C, H, W, dtype=...) ← Input is a tuple of two sparse tensors (B=1, C=3, H=10, W=3)
class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        z = x + y
        # Check if all values in the result are exactly 2.0 (correct case)
        correct = torch.all(z.to_dense() == 2.0)
        return correct

def my_model_function():
    return MyModel()

def GetInput():
    indices = torch.tensor([[7, 1, 3]])
    values_x = torch.tensor([[1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.]])
    values_y = torch.tensor(1.).expand(3, 3)  # Non-contiguous values to trigger the bug
    x = torch.sparse_coo_tensor(indices, values_x, size=(10, 3))
    y = torch.sparse_coo_tensor(indices, values_y, size=(10, 3))
    return (x, y)

