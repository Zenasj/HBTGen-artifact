import torch

types = [torch.uint8, torch.int8, torch.short, torch.int, torch.long, torch.half, torch.float, torch.double]

print('Testing linspace:')
for type in types:
    print(type, torch.linspace(-2, 2, 10, dtype=type))