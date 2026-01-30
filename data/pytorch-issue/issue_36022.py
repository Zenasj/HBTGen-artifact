import torch

idxs = torch.argmax(prob, dim=-1)
scores = torch.gather(prob, -1, idxs.unsqueeze(-1))