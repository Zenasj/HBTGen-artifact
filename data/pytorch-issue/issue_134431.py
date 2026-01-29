# torch.rand(2, 3, 0, 7, dtype=torch.float32)  # B=2, C=3, H=0, W=7
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, inputs):
        input, target = inputs
        try:
            loss_cpu = F.nll_loss(input.cpu(), target.cpu(), reduction='mean')
        except Exception:
            return torch.tensor(False, dtype=torch.bool)
        try:
            loss_mps = F.nll_loss(input.to('mps'), target.to('mps'), reduction='mean')
        except Exception:
            return torch.tensor(False, dtype=torch.bool)
        return torch.tensor(torch.allclose(loss_cpu, loss_mps, atol=1e-5), dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    input_size = [2, 3, 0, 7]
    num_channels = input_size[1]
    target_size = (input_size[0],) + tuple(input_size[2:])
    input = torch.rand(input_size, requires_grad=True, dtype=torch.float32)
    target = torch.randint(num_channels, target_size, dtype=torch.long)
    return input, target

