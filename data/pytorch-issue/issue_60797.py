import torch
from torch import nn
from warpctc_pytorch import CTCLoss

criterion_seq_decoder = CTCLoss()

# Case 1a: all ones, CPU
preds = torch.ones([32, 2, 211])
flatten_targets = torch.ones([41], dtype=torch.int32)
preds_size = torch.ones([32, 32], dtype=torch.int32)
length_for_loss = torch.ones([9, 32], dtype=torch.int32)

criterion_seq_decoder(
    preds, flatten_targets, preds_size, length_for_loss
)

# Case 1b: all ones, GPU
preds = torch.ones([32, 2, 211]).cuda()
flatten_targets = torch.ones([41], dtype=torch.int32)
preds_size = torch.ones([32, 32], dtype=torch.int32)
length_for_loss = torch.ones([9, 32], dtype=torch.int32)

criterion_seq_decoder(
    preds, flatten_targets, preds_size, length_for_loss
)

# Case 2a, loading input that led to NaN loss on GPU using CPU map_location
with open("dbg_ctc_th.pt", "rb") as buffer:
    dbg_info = torch.load(buffer, map_location=torch.device('cpu'))

dbg_info

preds = dbg_info['preds']
flatten_targets = dbg_info['flatten_targets']
preds_size = dbg_info['preds_size']
length_for_loss = dbg_info['length_for_loss']

# The CPU version for this input seems to be working
criterion_seq_decoder(
    preds, flatten_targets, preds_size, length_for_loss
)

# Rerun Case 1b: all ones, GPU, to show that cuda is still working
preds = torch.ones([32, 2, 211]).cuda()
flatten_targets = torch.ones([41], dtype=torch.int32)
preds_size = torch.ones([32, 32], dtype=torch.int32)
length_for_loss = torch.ones([9, 32], dtype=torch.int32)

criterion_seq_decoder(
    preds, flatten_targets, preds_size, length_for_loss
)

# Case 2b, loading input that led to NaN loss on GPU
with open("dbg_ctc_th.pt", "rb") as buffer:
    dbg_info = torch.load(buffer)

dbg_info

preds = dbg_info['preds']
flatten_targets = dbg_info['flatten_targets']
preds_size = dbg_info['preds_size']
length_for_loss = dbg_info['length_for_loss']

# The loss becomes 0 here, different from the CPU version!
criterion_seq_decoder(
    preds, flatten_targets, preds_size, length_for_loss
)

# Rerun Case 1b: all ones, GPU, to show that cuda stops working
preds = torch.ones([32, 2, 211]).cuda()
flatten_targets = torch.ones([41], dtype=torch.int32)
preds_size = torch.ones([32, 32], dtype=torch.int32)
length_for_loss = torch.ones([9, 32], dtype=torch.int32)