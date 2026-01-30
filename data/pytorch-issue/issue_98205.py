import torch

def diag2(x):
    diag_matrix = x.unsqueeze(1) * torch.eye(len(x)).to(x.device)
    return diag_matrix