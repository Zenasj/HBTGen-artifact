# torch.rand(B, 2, 2, dtype=torch.float32, device='cuda')  # B is the batch size, e.g., (1 << 16)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        a = x[:, [0], [0]]
        b = x[:, [1], [1]]
        s = (a + b).sum()
        return s

def my_model_function():
    return MyModel()

def GetInput():
    n = (1 << 16)  # Batch size
    x = torch.rand(n, 2, 2, dtype=torch.float32, device='cuda', requires_grad=True)
    return x

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)
# output.backward()
# assert torch.allclose(input_tensor.grad, torch.eye(2, out=input_tensor.new(2, 2)))

