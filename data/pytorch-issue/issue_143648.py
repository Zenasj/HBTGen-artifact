n = 0
input = torch.tensor(2, dtype=torch.float32)

import torch
import scipy

def polygamma(n, input):
    return torch.special.polygamma(n, input)

@torch.compile
def compiled_polygamma(n, input):
    return torch.special.polygamma(n, input)

n = 0
input = torch.tensor(2, dtype=torch.float32)
print(f"polygamma in Eager mode: ", polygamma(n, input))  # 0.4228
print(f"polygamma in compiled mode: ", compiled_polygamma(n, input))  # -inf
print(f"Scipy's result: ", scipy.special.polygamma(n, input.item()))  # 0.42278433509846713

import torch
print(torch.special.zeta(torch.tensor(0)+1, 2))  # inf