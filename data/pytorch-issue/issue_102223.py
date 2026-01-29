# torch.rand(2, dtype=torch.float32, requires_grad=True)
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        return checkpoint(self.run_fn, x, use_reentrant=True)
    
    def run_fn(self, x):
        create_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            y = torch.exp(x)
            z = y.sum()
            grads = torch.autograd.grad(z, x, create_graph=create_graph)[0]
        # Ensure the output has grad if needed (workaround from issue comments)
        return grads if grads.requires_grad else grads.clone().requires_grad_(True)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.float32, requires_grad=True)

