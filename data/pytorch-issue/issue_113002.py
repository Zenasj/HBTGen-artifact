import torch.nn as nn

import torch
from torch import nn

class PaddingNet(nn.Module):
    def __init__(self):
        super(PaddingNet, self).__init__()

    def forward(self, lengths):
        max_seq_len = lengths.max().item()
        row_vector = torch.arange(0, max_seq_len, 1)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix
        mask = mask.type(torch.float32)
        mask_3d_btd = mask[:, :, None]
        return mask_3d_btd

model = PaddingNet()
lengths = torch.tensor([5, 4, 4, 4], dtype=torch.int32)
with torch.no_grad():
    so_path, exported = torch._export.aot_compile(model, (lengths, ))

class PaddingNet(nn.Module):
    def __init__(self):
        super(PaddingNet, self).__init__()

    def forward(self, lengths):
        max_seq_len = lengths.max().item()
        torch._constrain_as_size(max_seq_len, max=5)  # I set this as max=5, but you can set it to a higher known value
        row_vector = torch.arange(0, max_seq_len, 1)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix
        mask = mask.type(torch.float32)
        mask_3d_btd = mask[:, :, None]
        return mask_3d_btd

model = PaddingNet()
lengths = torch.tensor([5, 4, 4, 4], dtype=torch.int32)
with torch.no_grad():
    exported_program = torch._export.export(model, (lengths,))