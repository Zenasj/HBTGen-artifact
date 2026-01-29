# torch.rand(B, 1, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Extract scalar condition from 4D tensor input
        predicate = x[0, 0, 0, 0] > 0  # Scalar condition check
        
        # Define HOO functions that process the full tensor input
        true_fn = lambda inputs: torch.sin(inputs[0])  # Operate on full tensor
        false_fn = lambda inputs: torch.cos(inputs[0])
        
        # Use Higher Order Operator with explicit mode handling
        return torch.ops.higher_order.cond(predicate, true_fn, false_fn, [x])[0]

def my_model_function():
    return MyModel()

def GetInput():
    # Generate 4D input tensor matching the model's expected input shape
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

