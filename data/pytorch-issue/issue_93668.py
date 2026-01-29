# torch.rand(10, 10, dtype=torch.float)  # GetInput returns a tuple of two tensors
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        z = x + y
        r = torch.relu(z)
        z_1 = self.gn(z, r) + 4
        r_1 = r + 4
        return torch.nn.functional.gelu(z_1 - r_1)
    
    def gn(self, x, y):
        torch.ops.profiler._record_function_enter("Starting the additions", "idk what this argument does")
        return x * y

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(10, 10), torch.rand(10, 10))

