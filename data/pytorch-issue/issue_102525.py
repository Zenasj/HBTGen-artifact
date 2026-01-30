import torch
x = torch.tensor([3+4j, 7-24j, 0, 1+2j])
print(torch.sgn(x)) # tensor([0.6000+0.8000j, 0.2800-0.9600j, 0.0000+0.0000j, 0.4472+0.8944j])

import torch

def torch_sgn_exp(x):
    return torch.where(x.real != 0, torch.sign(x.real) + 0.0j, torch.sign(x.imag) + 0.0j)

x = torch.tensor([3+4j, 7-24j, 0, 1+2j])
print(torch_sgn_exp(x)) # tensor([1.+0.j, 1.+0.j, 0.+0.j, 1.+0.j])