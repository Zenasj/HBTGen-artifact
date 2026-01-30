import torch
torch.set_default_device('cuda')

num_rows = 128
num_cols = 128
device='cuda'
def create_dense_one(kv_num_blocks, kv_indices):
    dense_mask = kv_indices.new_zeros(num_rows, num_cols + 1, dtype=torch.int32)

    row_indices = torch.arange(
        num_rows, dtype=torch.int, device=device
    ).unsqueeze(-1)
    col_indices = torch.arange(num_cols, dtype=torch.int, device=device)
    index_mask = col_indices < kv_num_blocks.unsqueeze(-1)

    # We write to one spot "out of bounds"
    valid_indices = torch.where(index_mask, kv_indices, num_cols)

    # set the values in 'a' to 1 where the indices are valid
    dense_mask[row_indices, valid_indices] = 1
    return dense_mask[:, :num_cols]

kv_num_blocks = torch.zeros(3, 128, device='cuda', dtype=torch.int)
kv_indices = torch.zeros(3, 128, 128, device='cuda', dtype=torch.int)
out = torch.vmap(create_dense_one, in_dims=(0, 0))(kv_num_blocks, kv_indices)