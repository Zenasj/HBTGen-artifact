import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import (
    BlockMask,
    _score_mod_signature,
    create_block_mask
)
from torch import nn

class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        dropout: float = 0.0,
        bias=False,
    ):
        super().__init__()
        assert (
            dim % n_head == 0
        ), f"dim must be divisible by n_head found: {dim} and {n_head}"
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.n_embd = dim
        self.dropout = dropout

    def forward(
        self,
        x,
        score_mod: None | _score_mod_signature = None,
        block_mask: None | BlockMask = None,
    ):
        B, T, C = (
            x.size()
        )  
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        ).contiguous()  ## Contiguous neccessary here https://github.com/pytorch/pytorch/issues/134471 TODO: Check if this is still necessary
        q, k, v = qkv

        y = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)


        return y


device = "cuda"
compile = True
dynamic = True
compile_block_mask = False

T = 256
E = 512
H = 8

assert T % 128 == 0, "T must be divisible by 128"

layer = SelfAttentionLayer(dim=E, n_head=H).to(device)
if compile:
    layer.compile(mode="default", dynamic=dynamic)
def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


block_mask = create_block_mask(
    mask_mod=causal,
    B=None,
    H=None,
    Q_LEN=T,
    KV_LEN=T,
    device=device,
    _compile=compile_block_mask,
)

for batch_size in range(2, 25):
    x = torch.randn(batch_size, T, E).to(device)
    y = layer(x,block_mask=block_mask)
    loss = y.mean()
    loss.backward()

import torch
from tqdm import tqdm
from torch.nn.attention.flex_attention import (
    BlockMask,
    _score_mod_signature,
    flex_attention,
    create_block_mask,
)
from torch import nn
import logging


torch._logging.set_logs(dynamo=logging.DEBUG)
torch._dynamo.config.verbose = True
torch.autograd.set_detect_anomaly(True)
torch._inductor.config.debug = True

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def batch_flip_causal(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & b % 2 == 0


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        dropout: float = 0.0,
        bias=False,
    ):
        super().__init__()
        assert (
            dim % n_head == 0
        ), f"dim must be divisible by n_head found: {dim} and {n_head}"

        # key, query, value projections for all heads, but in a batch
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        # regularization

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.head_dim = dim // n_head

        self.n_embd = dim
        self.dropout = dropout

    def forward(
        self,
        x,
        score_mod: None | _score_mod_signature = None,
        block_mask: None | BlockMask = None,
    ):
        B, T, C = (
            x.size()
        )  



        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  
        q, k, v = qkv

        y = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


@torch.no_grad()
def create_block_mask_from_dense(
    dense_mask: torch.Tensor, device, block_size: int = 128, compile: bool = False
) -> BlockMask:
    B, S = dense_mask.shape

    def mask_mod_partial(b, h, q_idx, kv_idx):
        return ~dense_mask[b, q_idx]

    return create_block_mask(
        B=B,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod_partial,
        H=None,
        Q_LEN=S,
        KV_LEN=S,
        _compile=compile,
        device=device,
    )


torch.set_float32_matmul_precision('high')
model = SelfAttentionLayer(
    dim = 512,
    n_head = 8,
    dropout = 0
).cuda()



compile_dynamic = True
compile_block_mask = True
sequence_len = 256
block_mask_type = "dense"



model.compile(mode="default", dynamic=compile_dynamic)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch_shape in tqdm(range(2,55)):
    
    match block_mask_type:
        case "causal":
            block_mask = create_block_mask(
            mask_mod=batch_flip_causal,
            B=None,
            H=None,
            Q_LEN=sequence_len,
            KV_LEN=sequence_len,
            device="cuda",
            _compile=compile_block_mask,
        )
        case "dense":        
            rand_mask = torch.randint(0,2,(batch_shape,sequence_len)).to("cuda").bool()
            block_mask = create_block_mask_from_dense(rand_mask,device="cuda",compile=compile_block_mask)
        case "batch_flip_causal":
            block_mask = create_block_mask(
                mask_mod=batch_flip_causal,
                B=batch_shape,
                H=None,
                Q_LEN=sequence_len,
                KV_LEN=sequence_len,
                device="cuda",
                _compile=compile_block_mask,
            )
        case _:
            raise ValueError("Invalid block_mask_type")
    x = torch.randn(batch_shape, sequence_len, 512).to("cuda")
    y = model(x,score_mod = None, block_mask = block_mask)
    loss = y.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()