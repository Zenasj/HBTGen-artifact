# torch.rand(B, 4, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

def bp_hook(module, grad_input, grad_output):
    param_grad = list(module.parameters())[0].grad
    print(f'gradient of the module inside bp_hook: {param_grad}')

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)
        self.linear.register_full_backward_hook(bp_hook)
        
    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 4, 1, 1, dtype=torch.float32)

