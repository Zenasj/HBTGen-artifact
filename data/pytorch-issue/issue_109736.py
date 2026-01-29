# torch.rand(B, T, dtype=torch.float16)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.old_mask = OldMaskModule()
        self.new_mask = NewMaskModule()

    def forward(self, x):
        mask_old = self.old_mask(x)
        mask_new = self.new_mask(x)
        # Compute cosine similarity between flattened masks
        flat_old = mask_old.view(-1)
        flat_new = mask_new.view(-1)
        sim = torch.nn.functional.cosine_similarity(flat_old, flat_new, dim=0)
        return sim

class OldMaskModule(nn.Module):
    def forward(self, x):
        tgt_len = x.size(1)
        dtype = x.dtype
        device = x.device
        mask_val = torch.tensor(torch.finfo(dtype).min, device=device)
        return torch.full((tgt_len, tgt_len), mask_val, device=device)

class NewMaskModule(nn.Module):
    def forward(self, x):
        tgt_len = x.size(1)
        dtype = x.dtype
        device = x.device
        mask_val = torch.finfo(dtype).min
        return torch.full((tgt_len, tgt_len), mask_val, device=device)

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size
    T = 10  # Target length
    return torch.rand(B, T, dtype=torch.float16)

