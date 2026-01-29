# torch.rand(1, dtype=torch.bool)  # Input shape: (1,), boolean tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, dilate_flag):
        # Extract condition from the first element of the boolean tensor
        condition = dilate_flag[0]
        
        def true_fn(x):
            return x.clone()

        def false_fn(x):
            return x.clone()  # Functions can be arbitrary; here matching original example

        # Execute torch.cond with symbolic boolean condition
        result = torch.cond(condition, true_fn, false_fn, (dilate_flag,))
        return result

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random boolean tensor of shape (1,)
    return torch.randint(0, 2, (1,), dtype=torch.bool)

