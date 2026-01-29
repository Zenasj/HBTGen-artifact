# torch.randint(0, 128, (128,), dtype=torch.int, device='cuda'), torch.randint(0, 128, (128, 128), dtype=torch.int, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_rows = 128
        self.num_cols = 128
        self.device = 'cuda'

    def forward(self, inputs):
        kv_num_blocks, kv_indices = inputs
        dense_mask = kv_indices.new_zeros(self.num_rows, self.num_cols + 1, dtype=torch.int32)
        row_indices = torch.arange(self.num_rows, dtype=torch.int, device=self.device).unsqueeze(-1)
        col_indices = torch.arange(self.num_cols, dtype=torch.int, device=self.device)
        index_mask = col_indices < kv_num_blocks.unsqueeze(-1)

        valid_indices = torch.where(index_mask, kv_indices, self.num_cols)

        dense_mask[row_indices, valid_indices] = 1
        return dense_mask[:, :self.num_cols]

def my_model_function():
    return MyModel()

def GetInput():
    kv_num_blocks = torch.randint(0, 128, (128,), dtype=torch.int, device='cuda')
    kv_indices = torch.randint(0, 128, (128, 128), dtype=torch.int, device='cuda')
    return (kv_num_blocks, kv_indices)

