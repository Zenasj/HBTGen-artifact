import torch.nn as nn

import code
import time

import warnings
import numpy as np
import torch
from torch.nn.attention.flex_attention import flex_attention, create_mask, create_block_mask

import astropy_healpix as hp


hlc = 3
num_healpix_cells = 12 * 4**hlc 
print( f'hlc={hlc}')
print( f'seq_length : {num_healpix_cells}')
num_heads = 8
dim_embed = 128
bs = 4

q = torch.ones( bs, num_heads, num_healpix_cells, dim_embed, dtype=torch.float16, device='cuda')
k = torch.ones( bs, num_heads, num_healpix_cells, dim_embed, dtype=torch.float16, device='cuda')
v = torch.ones( bs, num_heads, num_healpix_cells, dim_embed, dtype=torch.float16, device='cuda')

with warnings.catch_warnings(action="ignore"):
    nbours= hp.neighbours( np.arange(num_healpix_cells), 2**hlc, order='nested').transpose()
# build adjacency matrix (smarter ways to do it ...)
nbours_mat = torch.zeros( (num_healpix_cells,num_healpix_cells), dtype=torch.bool, device='cuda')
for i in range(num_healpix_cells) :
    for j in nbours[i] :
        nbours_mat[i,j] = True if j>=0 else False

# create sparse block matrix for flex attention
def sparse_mask(b, h, q_idx, kv_idx):
    # return ddkv_idx in nbours[q_idx]
    return nbours_mat[q_idx,kv_idx]
block_mask = create_block_mask( sparse_mask, B=None, H=None, Q_LEN=dim_embed, KV_LEN=dim_embed)

# experiments

# warmup
for i in range( 10):
  qp = flex_attention( q, k, v, block_mask=block_mask)

t_start = time.time()
for i in range( 1000):
  qp = flex_attention( q, k, v, block_mask=block_mask)
print( f'flex attention : {(time.time() - t_start) / 1000.} [s]', flush=True)

# warmup
for i in range( 10):
  with torch.nn.attention.sdpa_kernel( torch.nn.attention.SDPBackend.FLASH_ATTENTION) :
    qp = torch.nn.functional.scaled_dot_product_attention( q, k, v)

t_start = time.time()
for i in range( 1000):
  with torch.nn.attention.sdpa_kernel( torch.nn.attention.SDPBackend.FLASH_ATTENTION) :
    qp = torch.nn.functional.scaled_dot_product_attention( q, k, v)
print( f'dense attention : {(time.time() - t_start) / 1000.} [s]', flush=True)

import code
import time

import warnings
import numpy as np
import torch
from torch.nn.attention.flex_attention import flex_attention, create_mask, create_block_mask

import astropy_healpix as hp


hlc = 4
num_healpix_cells = 12 * 4**hlc
print( f'seq_length : {num_healpix_cells}')

with warnings.catch_warnings(action="ignore"):
    nbours= hp.neighbours( np.arange(num_healpix_cells), 2**hlc, order='nested').transpose()
# build adjacency matrix (smarter ways to do it ...)
nbours_mat = torch.zeros( (num_healpix_cells,num_healpix_cells), dtype=torch.bool, device='cuda')
for i in range(num_healpix_cells) :
    for j in nbours[i] :
        nbours_mat[i,j] = True if j>=0 else False
hp_adjacency = nbours_mat

# tc_tokens = torch.from_numpy( np.load( 'tc_tokens.npy')).to(torch.float16).to('cuda')
tc_tokens = torch.ones( [204458, 256], dtype=torch.float16, device='cuda', requires_grad=True)
tcs_lens = torch.from_numpy( np.load( './tcs_lens.npy')).to(torch.int32).to('cuda')
print( f'tc_tokens = {tc_tokens.shape}')
print( f'tcs_lens = {tcs_lens.shape}')

tc_tokens_cell_idx = torch.cat( [i * torch.ones( l, dtype=torch.int64, device='cuda')
                                                                for i,l in enumerate(tcs_lens)])
def sparsity_mask( score, b, h, q_idx, kv_idx):
    return hp_adjacency[ tc_tokens_cell_idx[q_idx], tc_tokens_cell_idx[kv_idx] ]

compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

toks = tc_tokens[:,:64].unsqueeze(0).unsqueeze(0)
out = compiled_flex_attention( toks, toks, toks, score_mod=sparsity_mask,
                               kernel_options={ 'BLOCK_M' : 64, 'BLOCK_N' : 64})

t = torch.zeros_like( out)
mse = torch.nn.MSELoss()
loss = mse( t, out)
loss.backward()

import warnings
import numpy as np
import torch
from torch.nn.attention.flex_attention import flex_attention

# -----------------------------
# Parameters and “Healpix” Setup
# -----------------------------
hlc = 4
num_healpix_cells = 12 * 4**hlc  # 12 * 256 = 3072
print(f'seq_length : {num_healpix_cells}')

# Instead of using astropy_healpix, we build an adjacency matrix for a 2D grid.
# For example, we choose a grid with dimensions (48, 64) because 48*64 = 3072.
num_rows, num_cols = 48, 64
if num_rows * num_cols != num_healpix_cells:
    raise ValueError("Chosen grid dimensions do not match the number of healpix cells.")

# Create an empty boolean adjacency matrix (on CUDA)
hp_adjacency = torch.zeros((num_healpix_cells, num_healpix_cells), dtype=torch.bool, device='cuda')

# Define neighbor offsets for an 8-connected grid (excluding self).
neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]

# Fill the adjacency matrix: for each cell, mark as True those cells that are valid neighbors.
for i in range(num_healpix_cells):
    row = i // num_cols
    col = i % num_cols
    for dr, dc in neighbor_offsets:
        nr = row + dr
        nc = col + dc
        if 0 <= nr < num_rows and 0 <= nc < num_cols:
            j = nr * num_cols + nc
            hp_adjacency[i, j] = True

# -----------------------------
# Dummy Tokens and Segment Lengths
# -----------------------------
# Create a dummy tokens tensor.
# (Originally loaded from file, now we simply make a tensor of ones.)
num_tokens = 204458
embed_dim = 256
tc_tokens = torch.ones([num_tokens, embed_dim],
                         dtype=torch.float16,
                         device='cuda',
                         requires_grad=True)

# Create a dummy tcs_lens vector.
# In the original code, tcs_lens was loaded from a file and its entries
# defined how many tokens belong to each healpix cell.
# Here we assume that each of the num_healpix_cells segments (cells) should sum up to num_tokens.
# For example, we can assign a base length and then distribute the remainder.
base_length = num_tokens // num_healpix_cells  # integer division
remainder = num_tokens - base_length * num_healpix_cells
tcs_lens_np = np.full((num_healpix_cells,), base_length, dtype=np.int32)
tcs_lens_np[:remainder] += 1  # distribute extra tokens to the first few segments
tcs_lens = torch.from_numpy(tcs_lens_np).to(torch.int32).to('cuda')

print(f'tc_tokens = {tc_tokens.shape}')
print(f'tcs_lens = {tcs_lens.shape}')

# For each token, assign the “cell index” that it comes from.
# That is, if tcs_lens = [l0, l1, l2, …] then tokens 0..l0-1 come from cell 0, next l1 tokens from cell 1, etc.
tc_tokens_cell_idx = torch.cat(
    [i * torch.ones(l, dtype=torch.int64, device='cuda') for i, l in enumerate(tcs_lens)]
)

# -----------------------------
# Sparsity Mask Function
# -----------------------------
# This function will be passed to flex_attention as a custom score modifier.
# It uses the token-to-cell mapping (tc_tokens_cell_idx) and the grid adjacency (hp_adjacency)
# to decide which query/key pairs are “allowed.”
def sparsity_mask(score, b, h, q_idx, kv_idx):
    # q_idx and kv_idx are indices into the tokens dimension.
    # We use them to look up the corresponding cell index and then return the precomputed connectivity.
    return score + hp_adjacency[tc_tokens_cell_idx[q_idx], tc_tokens_cell_idx[kv_idx]]

# -----------------------------
# Compile and Run Flex-Attention
# -----------------------------
# Optionally compile the flex_attention function for speed.
compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

# Prepare the tokens input.
# The original code takes the first 64 features of each token and adds two extra dimensions.
# (Often the extra dimensions are for batch and head.)
toks = tc_tokens[:, :64].unsqueeze(0).unsqueeze(0)  # shape: (1, 1, num_tokens, 64)

# Call flex_attention with the custom sparsity mask and some kernel options.
out = compiled_flex_attention(toks, toks, toks,
                              score_mod=sparsity_mask,
                              kernel_options={'BLOCK_M': 64, 'BLOCK_N': 64})

# -----------------------------
# Loss and Backward Pass
# -----------------------------
# For demonstration, compute a dummy loss (MSE to zeros) and run backprop.
t = torch.zeros_like(out)
mse = torch.nn.MSELoss()
loss = mse(t, out)
loss.backward()