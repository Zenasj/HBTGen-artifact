import torch
mask = torch.full((10, 10), float("-inf")).to("mps")
print(torch.triu(mask, diagonal=1))