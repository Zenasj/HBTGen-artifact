import torch.nn as nn

python # 
import torch

units = 1000
x = (torch.rand(1, units)).repeat(2,1)
weights = torch.rand(units,units)
result_single = torch.nn.functional.linear(x[:1], weights)
result_double = torch.nn.functional.linear(x[:2], weights)
# This will be True, x[0] is equal to x[1]
assert x[0].equal(x[1]), "inputs are different than one another"
# this will be True, result_double[0] is the same as result_double[1]
assert ((result_double[0]).equal(result_double[1])), 'outputs are different than one another for two identical samples in the same batch'
# this will be False, result_single[0] is different than result_double[0]
assert ((result_single[0]).equal(result_double[0])), 'outputs are different than one another for two identical samples in batches of different sizes'

python #
import torch
import matplotlib.pyplot as plt

errors = []
for units_pow in range(1,15):
    units = 2 ** units_pow
    x = (torch.rand(1, units)).repeat(2,1)
    weights = torch.rand(units,units)
    result_single = torch.nn.functional.linear(x[:1], weights)
    result_double = torch.nn.functional.linear(x[:2], weights)
    errors.append((result_double[0] - result_single[0]).abs().mean().item())

plt.plot(errors)
plt.xticks(range(14), [f'2^{i}' for i in range(1,15)], rotation=90)
plt.ylabel('Mean error')
plt.xlabel('Number of input and output units')