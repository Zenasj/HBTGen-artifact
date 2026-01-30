import torch.nn as nn

Python
import torch
import random
import numpy as np

import warnings

warnings.filterwarnings("ignore")

@torch.inference_mode()
def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    a, b = 1, 0.8
    batch_size = 128
    sequence_length = 128
    block_size = 32
    embed_dim = 1024
    num_heads = 16
    device = "cuda"
    dtype = torch.float16

    from scipy.stats import beta
    # lengths is ultimately just a list of numbers in the range [0,sequence_length] where each number is a multiple of block_size
    lengths = beta.rvs(a, b, size=batch_size) * (sequence_length + block_size) // block_size
    lengths_in_blocks = list(map(int, list(lengths)))
    lengths = [l * block_size for l in lengths_in_blocks]

    # Create q,k,v
    q = [torch.randn(l, embed_dim, device=device, dtype=dtype)for l in lengths]
    q = torch.nested_tensor(q, device=device, dtype=dtype)
    k, v = q, q

    qkv = torch.nn.Linear(embed_dim, 3 * embed_dim, device=device, dtype=dtype)
    proj = torch.nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

    native_mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, device=device, dtype=dtype).eval()
    native_mha.in_proj_weight = qkv.weight
    native_mha.in_proj_bias = qkv.bias
    native_mha.out_proj.weight = proj.weight
    native_mha.out_proj.bias = proj.bias

    y_native_mha, *rest = native_mha(q,k,v, need_weights=False)
    assert not torch.isnan(y_native_mha.to_padded_tensor(0)).any(), "I wouldn't expect NaNs to be present"

if __name__ == "__main__":
    main()

{
'batch:64_seq_len:64_n_heads:16_embed_dim:512': 121, 
'batch:64_seq_len:64_n_heads:16_embed_dim:1024': 121, 
'batch:128_seq_len:128_n_heads:16_embed_dim:512': 121, 
'batch:128_seq_len:128_n_heads:16_embed_dim:1024': 121, 
'batch:256_seq_len:256_n_heads:16_embed_dim:512': 121,
'batch:256_seq_len:256_n_heads:16_embed_dim:1024': 121
}

[96, 0, 0, 128, 96, 64, 96, 128, 64, 32, 128, 128, 128, 32, 0, 32, 64, 128, 64, 0, 96, 32, 0, 128, 0, 128, 128, 128, 64, 0, 96, 32, 0, 64, 128, 96, 128, 128, 32, 0, 96, 32, 96, 128, 32, 32, 32, 32, 0, 96, 0, 128, 96, 128, 96, 128, 32, 96, 0, 64, 128, 128, 0, 0, 128, 128, 32, 96, 64, 32, 128, 128, 0, 64, 32, 128, 32, 96, 64, 128, 128, 64, 32, 96, 32, 128, 128, 0, 96, 128, 128, 0, 96, 64, 96, 96, 0, 96, 128, 64, 32, 64, 128, 64, 128, 0, 0, 64, 64, 128, 128, 0, 32, 128, 0, 96, 0, 128, 128, 96, 64, 0, 0, 128, 0, 128, 0, 0]

@torch.inference_mode()
def test_transformerencoder_nans(force_seqlen_nonzero=False):
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    a, b = 1, 0.8
    batch_size = 128
    sequence_length = 128
    block_size = 32
    embed_dim = 1024
    num_heads = 16
    device = "cuda"
    dtype = torch.float16

    from scipy.stats import beta
    # lengths is ultimately just a list of numbers in the range [0,sequence_length] where each number is a multiple of block_size
    lengths = beta.rvs(a, b, size=batch_size) * (sequence_length + block_size) // block_size
    lengths_in_blocks = list(map(int, list(lengths)))
    lengths = [l * block_size for l in lengths_in_blocks]
    if(force_seqlen_nonzero):
        for i in range(len(lengths)):
            lengths[i] = max(lengths[i], 1)
    print(lengths)

    # Create q,k,v
    q = [torch.randn(l, embed_dim, device=device, dtype=dtype)for l in lengths]
    q = torch.nested_tensor(q, device=device, dtype=dtype)
    k, v = q, q

    native_mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, device=device, dtype=dtype).eval()

    y_native_mha, *rest = native_mha(q,k,v, need_weights=False)
    assert not torch.isnan(y_native_mha.to_padded_tensor(0)).any(), "I wouldn't expect NaNs to be present"