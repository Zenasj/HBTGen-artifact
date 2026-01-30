import torch.nn as nn

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
num_heads = 8
dim_embed = 64
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
hp_adjacency = nbours_mat

# tc_tokens = torch.from_numpy( np.load( 'tc_tokens.npy')).to(torch.float16).to('cuda')
tc_tokens = torch.ones( [204458, 256], dtype=torch.float16, device='cuda')
tcs_lens = torch.from_numpy( np.load( 'tcs_lens.npy')).to(torch.int32).to('cuda')
print( f'tc_tokens = {tc_tokens.shape}')
print( f'tcs_lens = {tcs_lens.shape}')

tc_tokens_cell_idx = torch.cat( [i * torch.ones( l, dtype=torch.int64) 
                                                                for i,l in enumerate(tcs_lens)])
def sparsity_mask( score, b, h, q_idx, kv_idx):
    return hp_adjacency[ tc_tokens_cell_idx[q_idx], tc_tokens_cell_idx[kv_idx] ]

compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

# poor mans head projection
toks = tc_tokens[:,:64].unsqueeze(0).unsqueeze(0)
out = compiled_flex_attention( toks, toks, toks, score_mod=sparsity_mask)

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
out = compiled_flex_attention( toks, toks, toks, score_mod=sparsity_mask)

t = torch.zeros_like( out)
mse = torch.nn.MSELoss()
loss = mse( t, out)
loss.backward()