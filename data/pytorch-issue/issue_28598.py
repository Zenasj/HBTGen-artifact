import torch
self = torch.tensor([1, 2, 3, 4, 5, 6, 7])
mask = torch.tensor([1, 0, 0, 1, 0, 1, 1], dtype=torch.bool)
src = torch.tensor([100, 200, 300, 400]) # src entry is picked up just "from left to right"

self.masked_scatter(mask, src)
# => tensor([100,   2,   3, 200,   5, 300, 400])