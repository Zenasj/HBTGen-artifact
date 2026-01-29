# torch.rand(5, device='cuda'), torch.rand(2)  # Input: (CUDA tensor, CPU tensor)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        main_tensor, value = inputs
        indices = torch.tensor([0, 1], device=main_tensor.device)
        main_tensor.index_put_((indices,), value, accumulate=True)
        return main_tensor

def my_model_function():
    return MyModel()

def GetInput():
    main_tensor = torch.rand(5, device='cuda')
    value = torch.rand(2)  # CPU by default
    return (main_tensor, value)

