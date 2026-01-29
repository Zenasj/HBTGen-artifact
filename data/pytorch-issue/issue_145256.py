# torch.nested.nested_tensor([...], layout=torch.jagged)  # batch_size=4, varying lengths, dim=64
import torch
import torch.nn as nn

def modulate(x, shift, scale):
    if scale is not None:
        x = x * (1 + scale)
    if shift is not None:
        x = x + shift
    return x

class MyModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.modulation = nn.Linear(dim, 2 * dim)

    def forward(self, inputs):
        x, c = inputs
        shift, scale = self.modulation(c).chunk(2, dim=-1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        return modulate(x, shift, scale)

def my_model_function():
    return MyModel(64)

def GetInput():
    tensors = [torch.randn(64, 64), torch.randn(128, 64), 
               torch.randn(256, 64), torch.randn(512, 64)]
    x = torch.nested.nested_tensor(tensors, layout=torch.jagged, requires_grad=True)
    c = torch.randn(4, 64, requires_grad=True)
    return (x, c)

