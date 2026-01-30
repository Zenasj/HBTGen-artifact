import torch

3
preds = torch.ones(5, 68, 64, 64) * 0.1
preds.sum() * 10

3
preds = torch.ones(5, 68, 64, 64)
preds.sum()

preds = torch.ones(5, 68, 64, 64, dtype=torch.float64) * 0.1
preds.sum() * 10