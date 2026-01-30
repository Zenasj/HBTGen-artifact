import torch
print(torch.tril_indices(4,2).dtype)
print(torch.tril_indices(4,2, device=torch.device("cuda")).dtype)