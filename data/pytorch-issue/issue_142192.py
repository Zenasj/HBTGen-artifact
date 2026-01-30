import torch

a = torch.tensor([1.0, 2, 1.0, 2.5, 3.5], dtype=torch.bfloat16)
unique, counts = torch.unique(a.to('cuda'), return_counts=True)