import torch

mask = torch.full(
    (10, 10), float("-inf"), device="mps"
)
print(mask)
mask = torch.triu(mask, diagonal=1)
print(mask)