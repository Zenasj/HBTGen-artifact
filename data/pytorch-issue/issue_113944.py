import torch

def add(tensor, scalar, alpha):
  return torch.add(tensor, scalar, alpha=alpha)

input_tensor = torch.randint(low = 10, high = 50, size = (4,5), dtype = torch.int)
compiled_add = torch.compile(add)
scalar = 7
alpha = 3
compiled_add(input_tensor, scalar, alpha)