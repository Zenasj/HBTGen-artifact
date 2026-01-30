import torch.nn as nn

import torch
from typing import List, Union
from torch import Tensor
from torch.nn.attention.flex_attention import create_mask

def _offsets_to_doc_ids_tensor(offsets):
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )

def length_to_offsets(lengths: List[int], device: Union[str, torch.device]) -> Tensor:
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets

def generate_doc_mask_mod(offsets: Tensor):
    document_id = _offsets_to_doc_ids_tensor(offsets)
    def doc_mask_mod(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]
    return doc_mask_mod

def main(device: str = "cpu"):
    import random
    random.seed(0)

    def generate_random_lengths(total_length, num_documents):
        lengths = [1] * num_documents
        remaining_length = total_length - num_documents
        for _ in range(remaining_length):
            index = random.randint(0, num_documents - 1)
            lengths[index] += 1
        return lengths

    max_seq_len, doc_count = 128, 4
    B, H, SEQ_LEN, HEAD_DIM = 1, 1, max_seq_len, 8

    lengths = generate_random_lengths(max_seq_len, doc_count)
    offsets = length_to_offsets(lengths, device)

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()
    document_causal_mask = generate_doc_mask_mod(offsets)

    # Error occurs when `_compile=True` is set
    mask = create_mask(document_causal_mask, 1, 1, SEQ_LEN, SEQ_LEN, device=query.device, _compile=True)

main()