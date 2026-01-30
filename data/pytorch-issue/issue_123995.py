import torch.nn as nn

import torch

input=torch.rand(32, 32)
assert input.device.type == "cpu"

weight=torch.randint(-100, 100, (32, 32), dtype=torch.int8)
assert weight.device.type == "cpu"

print(torch.nn.functional.linear(input, weight.to(dtype=input.dtype)))

import torch

input=torch.rand(32, 32, device='mps')
assert input.device.type == "mps"
weight=torch.randint(-100, 100, (32, 32), dtype=torch.int8, device='mps')
assert weight.device.type == "mps"

print(torch.nn.functional.linear(input, weight.to(dtype=input.dtype)))