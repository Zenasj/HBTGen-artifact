import torch

indices = (torch.tensor([0, 1, 2]), torch.tensor([2, 0, 1]))
values = torch.tensor([0., 3., 4.])
a = torch.zeros(size=(5, 5))
# Works.
a.index_put_(indices=indices, values=values)

# Fails.
a.index_put_(indices=indices, value=values)