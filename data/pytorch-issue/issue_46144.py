# torch.rand(1, dtype=torch.complex64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.re_weights = nn.Linear(1, 1)
        self.im_weights = nn.Linear(1, 1)
        
    def forward(self, x):
        multiplied = torch.view_as_complex(
            torch.stack([self.re_weights(x.real), self.im_weights(x.imag)], dim=-1)
        )
        angle_atan2 = multiplied.imag.atan2(multiplied.real)
        angle_torch = multiplied.angle()
        return angle_atan2, angle_torch  # Return both angle computations for comparison

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.complex64)

