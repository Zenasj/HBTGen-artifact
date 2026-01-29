# torch.rand(4, 2, dtype=torch.float32)  # Input is a tensor with x (requires_grad) and y (no grad) along the second dimension
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mask', torch.BoolTensor([True, False, True, True]))
    
    def forward(self, inputs):
        x = inputs[:, 0]  # Extract x values (requires_grad)
        y = inputs[:, 1].detach()  # y is treated as constant
        
        # Case 1: Mask before atan2
        masked_x = x[self.mask]
        masked_y = y[self.mask]
        out_before = torch.atan2(masked_y, masked_x).mean()
        
        # Case 2: Mask after atan2
        atan2_all = torch.atan2(y, x)
        out_after = atan2_all[self.mask].mean()
        
        return out_before, out_after  # Return both outputs for gradient comparison

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(4, dtype=torch.float32, requires_grad=True)  # Only x requires gradient
    y = torch.rand(4, dtype=torch.float32, requires_grad=False)
    inputs = torch.stack([x, y], dim=1)  # Shape (4,2)
    return inputs

