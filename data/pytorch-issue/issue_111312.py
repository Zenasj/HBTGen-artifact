import torch.nn as nn

import torch
from functorch import make_fx
from torch.func import vmap, jacrev

trace_inp = torch.randn(1, 2)

def model(x):
    numel = (x.numel(),)
    return torch.split_with_sizes(x, numel)

# Used to fail with 
# RuntimeError: isIntList() INTERNAL ASSERT FAILED at "/home/git/pytorch/aten/src/ATen/core/ivalue_inl.h":1968,
# please report a bug to PyTorch. Expected IntList but got GenericList
make_fx(vmap(model), tracing_mode='symbolic')(trace_inp)

# User to fail with
#   File "/home/git/pytorch/torch/_functorch/eager_transforms.py", line 117, in _autograd_grad
#     grad_inputs = torch.autograd.grad(diff_outputs, inputs, grad_outputs,
#   File "/home/git/pytorch/torch/autograd/__init__.py", line 372, in grad
#     grad_outputs_ = _make_grads(
#   File "/home/git/pytorch/torch/autograd/__init__.py", line 68, in _make_grads
#     if not torch.is_same_size(out, first_grad):
# RuntimeError: Cannot call sizes() on tensor with symbolic sizes/strides
model = torch.nn.Tanh()
make_fx(jacrev(model), tracing_mode='symbolic')(trace_inp)

import torch
from functorch import make_fx
from torch.func import vmap, jacrev

model = torch.nn.Sequential(
    torch.nn.Linear(2, 512),
    torch.nn.Tanh(),
    torch.nn.Linear(512, 2),
)

trace_inp = torch.randn(1, 2)
traced_model = make_fx(vmap(jacrev(model)), tracing_mode='symbolic',
                       _allow_non_fake_inputs=True)(trace_inp)

# This will work
print(traced_model(torch.randn(1, 2)))

# This will fail with: RuntimeError: shape '[1, 2]' is invalid for input of size 4
print(traced_model(torch.randn(2, 2)))