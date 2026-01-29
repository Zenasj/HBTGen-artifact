# torch.rand(3, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, input_tensor):
        start = input_tensor[0].item()
        end = input_tensor[1].item()
        steps = int(input_tensor[2].item())
        cpu_out = torch.logspace(start, end, steps, device='cpu', dtype=torch.int32)
        cuda_out = torch.logspace(start, end, steps, device='cuda', dtype=torch.int32)
        # Return boolean comparison as a tensor
        return torch.tensor([torch.equal(cpu_out.to('cuda'), cuda_out)], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Parameters from the example in the issue (start=0, end=3, steps=4)
    return torch.tensor([0.0, 3.0, 4.0], dtype=torch.float)

