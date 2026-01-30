import torch
print(torch.arange(9).reshape(3, 3).to("cpu")[[1, 2], [2, 1]])
print(torch.arange(9).reshape(3, 3).to("mps")[[1, 2], [2, 1]])