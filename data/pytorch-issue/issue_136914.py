import torch
import torch.nn as nn
from torch import Tensor
from triton.testing import do_bench
from typing import Any, List, Optional, Tuple, Union
from torch.nn.attention.flex_attention import BlockMask, _mask_mod_signature, _score_mod_signature, flex_attention, create_block_mask
import random

torch.set_default_device('cuda')

def _offsets_to_doc_ids_tensor(offsets):
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )


def length_to_offsets(lengths: List[int], device: Union[str, torch.device]) -> Tensor:
    """Converts a list of lengths to a list of offsets.

    Args:
        lengths: A list of lengths.

    """
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets


def generate_doc_mask_mod(mask_mod: _mask_mod_signature, offsets: Tensor) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        offsets: This tensor should be of shape(num_documents + 1)
            this should contain the cumulative counts of document tokens.
            e.g. if you have 3 documents of length 2, 4, 3 then
            offsets = [0, 2, 6, 9]

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.
    """
    document_id = _offsets_to_doc_ids_tensor(offsets)

    def doc_mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        q_logical = q_idx - offsets[document_id[q_idx]]
        kv_logical = kv_idx - offsets[document_id[kv_idx]]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask

    return doc_mask_mod

def generate_random_lengths(total_length, num_documents):
    # Initialize all lengths to 1 to ensure each document has at least one token
    lengths = [1] * num_documents
    remaining_length = total_length - num_documents

    # Randomly distribute the remaining length
    for _ in range(remaining_length):
        index = random.randint(0, num_documents - 1)
        lengths[index] += 1

    return lengths

for i in range(10):
    lengths = generate_random_lengths(1024 + i, 5)
    offsets = length_to_offsets(lengths, 'cuda')
    doc_ids = _offsets_to_doc_ids_tensor(offsets)
    total_seq_len = 1024 + i
    
    def doc_mask_mod(b, h, q_idx, kv_idx):
        return (
            doc_ids[q_idx.clamp(0, doc_ids.shape[0] - 1)]
            == doc_ids[kv_idx.clamp(0, doc_ids.shape[0] - 1)]
        )
    q, k, v = [torch.randn(1, 12, 1024 + i, 64) for _ in range(3)]
    block_mask = create_block_mask(doc_mask_mod, None, None, 1024 + i, 1024 + i)
    torch.compile(flex_attention)(q, k, v, block_mask=block_mask)