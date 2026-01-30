import torch.nn as nn

py
import torch
torch.manual_seed(0)

def f(input_data, device='cpu'):
    model = torch.nn.InstanceNorm3d(3, device=device)
    out = model(input_data)
    return out.sum()

input_data = torch.rand(100, 3, 1, 10, 10)

inp_cpu = input_data.clone().cpu()
inp_cuda = input_data.clone().cuda()

print(f(inp_cpu)) # tensor(0.0001)
print(f(inp_cuda, 'cuda')) # tensor(0.0014, device='cuda:0')