import torch
source=torch.ones(91,91)
target=torch.zeros(91,91)
target.add_(source)
print(target)