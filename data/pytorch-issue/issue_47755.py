import torch

py
inputs = torch.randn(N, 3)
inputs_clone = inputs.clone()
f = lambda x: x.fill_(0)
[f(inputs[i]) for i in range(N)] # zero-fill each slice of inputs
expected = torch.vmap(f)(inputs_clone)    # zero-fill all slices of inputs.
assert torch.allclose(inputs, expected)