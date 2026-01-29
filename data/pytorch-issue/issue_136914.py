# (torch.rand(1, 12, 1024, 64, dtype=torch.float32, device='cuda'), ) *3 

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import random
from typing import List, Union, Tuple

def generate_random_lengths(total_length: int, num_documents: int) -> List[int]:
    lengths = [1] * num_documents
    remaining = total_length - num_documents
    for _ in range(remaining):
        idx = random.randint(0, num_documents-1)
        lengths[idx] += 1
    return lengths

def length_to_offsets(lengths: List[int], device: Union[str, torch.device]) -> Tensor:
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    return torch.cumsum(offsets, dim=0)

def _offsets_to_doc_ids_tensor(offsets: Tensor) -> Tensor:
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32),
        counts
    )

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        q, k, v = inputs
        seq_len = q.shape[2]
        lengths = generate_random_lengths(seq_len, 5)
        offsets = length_to_offsets(lengths, q.device)
        doc_ids = _offsets_to_doc_ids_tensor(offsets)
        
        def doc_mask_mod(b, h, q_idx, kv_idx):
            return (doc_ids[q_idx.clamp(0, doc_ids.shape[0]-1)] == 
                    doc_ids[kv_idx.clamp(0, doc_ids.shape[0]-1)])
        
        block_mask = create_block_mask(doc_mask_mod, None, None, seq_len, seq_len)
        return flex_attention(q, k, v, block_mask=block_mask)

def my_model_function():
    return MyModel()

def GetInput():
    q = torch.rand(1, 12, 1024, 64, dtype=torch.float32, device='cuda')
    k = torch.rand(1, 12, 1024, 64, dtype=torch.float32, device='cuda')
    v = torch.rand(1, 12, 1024, 64, dtype=torch.float32, device='cuda')
    return (q, k, v)

