import torch

mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)

mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)