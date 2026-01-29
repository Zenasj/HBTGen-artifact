# torch.rand(B, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class DummyTensor(torch.Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__()

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)
        # Register the backward hook causing type change
        def backward_hook(module, grad_input, grad_output):
            return  # do nothing
        self.register_full_backward_hook(backward_hook)
    
    def forward(self, x):
        return DummyTensor(self.fc(x))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 2, dtype=torch.float32)

