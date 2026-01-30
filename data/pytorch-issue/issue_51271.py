import torch

combined = torch.cat((t1, t2))
uniques, counts = combined.unique(return_counts=True)
difference = uniques[counts == 1]
intersection = uniques[counts > 1]