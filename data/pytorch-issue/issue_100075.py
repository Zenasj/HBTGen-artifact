# torch.rand(10, 3, 224, 224, dtype=torch.float32)  # Inferred input shape

import torch
from torch import nn
from torchvision.models import resnet18
from torch._dynamo import allow_in_graph
from functools import wraps
from functorch import make_functional_with_buffers, vmap, grad

def traceable(f):
    f = allow_in_graph(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = resnet18(pretrained=False, norm_layer=(lambda c: nn.GroupNorm(min(c, 32), c)))
        self.model.eval()

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    device = 'cpu'  # also fails on CUDA
    x = torch.randn(10, 3, 224, 224, device=device)
    return x

