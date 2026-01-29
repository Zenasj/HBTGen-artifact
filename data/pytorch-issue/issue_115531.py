# torch.rand(3, 3, 7, 7, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Adjusted kernel_size to 3 (valid per the constraint) while keeping other parameters from the example
        self.pool = nn.FractionalMaxPool2d(
            kernel_size=3,  # Must be < half the input's H/W (7 → 3.5 → 3 is valid)
            output_ratio=0.8450560840300386,
            return_indices=False
        )
    
    def forward(self, x):
        return self.pool(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 3, 7, 7, dtype=torch.float32)

