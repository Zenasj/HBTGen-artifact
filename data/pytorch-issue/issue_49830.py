# torch.rand(10, 5, requires_grad=True) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Initialize a learnable parameter
        self.tmp = nn.Parameter(torch.randn(10, 5, requires_grad=True))

    def forward(self, x):
        return x.sum()

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(10, 5, requires_grad=True)

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)
# jacobian = torch.autograd.functional.jacobian(model, input_tensor)
# print(jacobian)

