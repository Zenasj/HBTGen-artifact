import torch

device = 'cpu'
# device = 'cuda'
input = torch.zeros(4, 4, device=device)
src = torch.ones(2, 2, device=device)
index = torch.tensor([[1, 3], [2, 0]], device=device, dtype=torch.long)
input.scatter_add_(0, index, src)
print(input)