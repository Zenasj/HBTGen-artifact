# torch.rand(4, 10, dtype=torch.float32)  # Inferred input shape from the provided code

import torch
from torch.utils.checkpoint import checkpoint

class MyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ln1 = torch.nn.Linear(10, 20)
        self.ln2 = torch.nn.Linear(20, 3000)
        self.ln3 = torch.nn.Linear(3000, 40)

    def forward(self, x):
        x = self.ln1(x)
        x = torch.nn.functional.relu(x)
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x.detach(), lambda x: x):
            x = self.ln2(x)
            x = torch.nn.functional.relu(x)
        x = self.ln3(x)
        x = torch.nn.functional.relu(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(4, 10, device='cuda')

