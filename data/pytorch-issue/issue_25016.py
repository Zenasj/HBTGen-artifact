import torch
x = torch.ones(3)
y = torch.ones(3) * torch.tensor(float('nan'))

print(f'torch.max(x, y) : {torch.max(x, y)}')
print(f'torch.max(y, x) : {torch.max(y, x)}')

a = torch.tensor([float('nan'), 1])
b = torch.tensor([1, float('nan')])
a.max()
b.max()