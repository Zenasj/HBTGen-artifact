import torch
from torch import nn

class CustomDict(dict):
    # Placeholder for a custom dictionary class that Dynamo can't handle
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Dynamo doesn't support custom dict inits with args/kwargs
        pass

# torch.rand(B, C, H, W, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # This line triggers the Dynamo error (custom dict init with args/kwargs)
        custom_dict = CustomDict(x.shape, some_kwarg=True)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

