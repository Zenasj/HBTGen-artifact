import torch

mask = torch.full((max_seq_length, max_seq_length), float("-inf"), device=device)
mask = torch.triu(mask, diagonal=1)
if str(device).startswith("mps"):
    mask = torch.nan_to_num(mask, nan=0.0)