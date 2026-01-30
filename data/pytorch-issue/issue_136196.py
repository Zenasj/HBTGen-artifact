import torch.nn as nn

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

try:
    import torch
    from torch.nn.attention.flex_attention import (
        flex_attention,
    )

    flex_attention = torch.compile(flex_attention)
    print(f"Torch version: {torch.__version__}")
except ImportError:
    flex_attention = None


def create_document_score_mod(sequence_ids):
    def scorer(score, b, h, q_idx, kv_idx):
        return torch.where(sequence_ids[b, q_idx] == sequence_ids[b, kv_idx], score, torch.finfo(score.dtype).min)

    return scorer

batch_sizes = [4, 8, 16, 17, 19, 20]
sequence_lengths = [2048, 3087]

for batch_size in batch_sizes:
    for sequence_length in sequence_lengths:
        sequence_ids = torch.randint(0, 10, (batch_size, sequence_length)).to("cuda")
        score_mod = create_document_score_mod(sequence_ids)

        q = torch.randn(batch_size, 8, sequence_length, 64).to("cuda")
        k = torch.randn(batch_size, 8, sequence_length, 64).to("cuda")
        v = torch.randn(batch_size, 8, sequence_length, 64).to("cuda")

        y = flex_attention(q, k, v, score_mod=score_mod)
        print(y.shape)