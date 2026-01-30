import torch.nn as nn
import torchvision

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

torch.manual_seed(42)
device = 'cpu'  # also fails on CUDA

model = resnet18(pretrained=False, norm_layer=(lambda c: nn.GroupNorm(min(c, 32), c)))
model.to(device)
model.eval()

fnet, params, buffers = make_functional_with_buffers(model)

x = torch.randn(10, 3, 224, 224, device=device)
f = lambda p, b, x : fnet(p, b, x).sum()

# Works for this simpler function
# f = lambda p, b, x: (torch.sin(x) + torch.cos(x) + torch.exp(x)).sum()

f = grad(f)

expected = f(params, buffers, x)
actual = torch.compile(traceable(f))(params, buffers, x)

torch.testing.assert_close(actual, expected)

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

torch.manual_seed(42)
device = 'cpu'  # also fails on CUDA

model = resnet18(pretrained=False)
model.to(device)
model.eval()

fnet, params, buffers = make_functional_with_buffers(model)

x = torch.randn(10, 3, 224, 224, device=device)
f = lambda p, b, x : fnet(p, b, x).sum()

# Works for this simpler function
# f = lambda p, b, x: (torch.sin(x) + torch.cos(x) + torch.exp(x)).sum()

f = grad(f)

expected = f(params, buffers, x)
actual = torch.compile(traceable(f))(params, buffers, x)

torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torch._dynamo import allow_in_graph
from functools import wraps
from functorch import make_functional_with_buffers, vmap, grad

def traceable(f):
    f = allow_in_graph(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper

torch.manual_seed(42)
device = 'cpu'  # also fails on CUDA

model = resnet18(pretrained=ResNet18_Weights.DEFAULT)
model.to(device)
model.eval()

fnet, params, buffers = make_functional_with_buffers(model)

x = torch.randn(10, 3, 224, 224, device=device)
f = lambda p, b, x : fnet(p, b, x).sum()

# Works for this simpler function
# f = lambda p, b, x: (torch.sin(x) + torch.cos(x) + torch.exp(x)).sum()

f = grad(f)

expected = f(params, buffers, x)
actual = torch.compile(traceable(f))(params, buffers, x)

torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)