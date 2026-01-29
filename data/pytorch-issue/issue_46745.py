import torch
import numpy as np

# torch.rand(B, 10, dtype=torch.float32).cuda()
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('cplx', torch.from_numpy(1j * np.arange(10)))

    def forward(self, x):
        return x * self.cplx

def my_model_function():
    model = MyModel()
    model.cuda()  # Ensure model is on CUDA as per the original issue's context
    return model

def GetInput():
    return torch.rand(2, 10).cuda()  # Matches input shape used in the original reproduction code

