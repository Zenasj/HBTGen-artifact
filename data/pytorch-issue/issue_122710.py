import torch

py
def print_tensor(t):
    print(t.flatten())

nan_cpu = torch.ones(2, device='cpu') * float('nan')
nan_mps = torch.ones(2, device='mps') * float('nan')

print_tensor(nan_cpu)  # tensor([nan, nan])
print_tensor(nan_cpu.clamp(0, 1)) # tensor([nan, nan])

print_tensor(nan_mps)  # tensor([nan, nan], device='mps:0')
print_tensor(nan_mps.clamp(0, 1)) # tensor([0., 0.], device='mps:0')