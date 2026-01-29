# torch.rand(1, 100, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Submodules representing the two cases from the issue (integer vs float scale_factor)
        self.upsample_int = nn.Upsample(scale_factor=2)
        self.upsample_float = nn.Upsample(scale_factor=2.5)
    
    def forward(self, x):
        # Apply both upsample operations and return outputs (as per comparison in the issue)
        out_int = self.upsample_int(x)
        out_float = self.upsample_float(x)
        # Return both outputs to demonstrate behavior differences (issue's comparison context)
        return out_int, out_float

def my_model_function():
    # Returns the fused model containing both upsample variants
    return MyModel()

def GetInput():
    # Generate input matching the 3D shape from the issue's reproduction code
    return torch.rand(1, 100, 100, dtype=torch.float32)

