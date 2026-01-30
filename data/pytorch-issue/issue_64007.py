import torch.nn as nn

import torch
import torch.nn.functional as F

input = torch.rand(128, 8)
weight = torch.rand(10, 4)
bias = torch.rand(1)

def run_linear_cpu(input, weight, bias):
    return F.linear(input, weight, bias)

def run_linear_gpu(input, weight, bias):
    return F.linear(input.cuda(), weight.cuda(), bias.cuda())

print(f"input: {input.size()}; weight: {weight.size()}; bias: {bias.size()}")

# GPU version: (shouldn't have passed)
print(f"gpu result: {run_linear_gpu(input, weight, bias).size()}")

# CPU version: (fail as expected)
print(f"cpu result: {run_linear_cpu(input, weight, bias).size()}")