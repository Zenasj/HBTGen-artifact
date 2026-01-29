# torch.rand(5, 5, 5, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        indices = [1, 2]
        # Compare instance-based __getitem__ vs class-based __getitem__
        instance_result = x.__getitem__(indices)
        class_result = torch.Tensor.__getitem__(x, indices)
        # Check if outputs are identical (same values and not views)
        return torch.allclose(instance_result, class_result) and torch.all(instance_result == class_result)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 5, 5, dtype=torch.float32)

