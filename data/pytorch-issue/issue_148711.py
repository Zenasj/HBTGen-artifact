import torch.nn as nn

attn_mask = torch.bernoulli(
    torch.zeros([batch_size, 1, seq_length, seq_length]) + 0.5, generator=torch.Generator().manual_seed(42)
) * torch.finfo(torch.float32).min
attn_mask = attn_mask.to(device)

if flex:
            head_max = torch.tensor(attn_mask.shape[1] - 1, device=attn_mask.device)
            attn_output = flex_compiled(  # [B, num_heads, L, head_dim]
                query_states,
                key_states,
                value_states,
                score_mod=(
                    lambda score, batch, head, q_idx, k_idx: (
                        score + attn_mask[batch, torch.min(head_max, head), q_idx, k_idx]
                    )
                )
            )

import torch
import numpy as np
from typing import Optional, Tuple

from torch.nn.attention.flex_attention import flex_attention

flex_compiled = torch.compile(flex_attention, mode ="max-autotune-no-cudagraphs", fullgraph=True)

class MultiheadAttention(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        out_features: Optional[int] = None,
        key_features: Optional[int] = None,
        value_features: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.key_features = key_features or in_features
        self.value_features = value_features or in_features
        self.out_features = out_features or in_features

        self.num_heads = num_heads
        self.head_dim = head_dim or in_features // num_heads
        self.num_kv_heads = num_kv_heads or num_heads

        self.q_proj = torch.nn.Linear(self.in_features, self.num_heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(self.key_features, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(self.value_features, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(self.num_heads * self.head_dim, self.out_features, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        flex: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = query.shape[0]

        # Input projection
        query_states = self.q_proj(query)  # [B, L, num_heads * head_dim]
        key_states = self.k_proj(key)  # [B, S, num_kv_heads * head_dim]
        value_states = self.v_proj(value)  # [B, S, num_kv_heads * head_dim]

        # Reshape
        query_states = query_states.view(  # [B, num_heads, L, head_dim]
            batch_size, query_states.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(  # [B, num_kv_heads, S, head_dim]
            batch_size, key_states.shape[1], self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(  # [B, num_kv_heads, S, head_dim]
            batch_size, value_states.shape[1], self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        key_states = expand_kv_heads(  # [B, num_heads, S, head_dim]
            key_states, self.num_heads // self.num_kv_heads
        )
        value_states = expand_kv_heads(  # [B, num_heads, S, head_dim]
            value_states, self.num_heads // self.num_kv_heads
        )

        if flex:
            head_max = torch.tensor(attn_mask.shape[1] - 1, device=attn_mask.device)
            attn_output = flex_compiled(  # [B, num_heads, L, head_dim]
                query_states,
                key_states,
                value_states,
                score_mod=(
                    lambda score, batch, head, q_idx, k_idx: (
                        score + attn_mask[batch, torch.min(head_max, head), q_idx, k_idx]
                    )
                )
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(  # [B, num_heads, L, head_dim]
                query_states,
                key_states,
                value_states,
                attn_mask=attn_mask,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, L, num_heads, head_dim]
        attn_output = attn_output.view(  # [B, L, num_heads * head_dim]
            batch_size, query.shape[1], self.num_heads * self.head_dim
        )

        attn_output = self.o_proj(attn_output)  # [B, L, out_features]

        return attn_output, None

def expand_kv_heads(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    The equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). Convert hidden_states from
        [batch, num_kv_heads, seqlen, head_dim] -> [batch, num_attention_heads, seqlen, head_dim]
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)



num_trials = 50
depth = 18
seq_length = 320
batch_size = 128
device = 'cuda'
dtype = torch.bfloat16

attn = MultiheadAttention(
    in_features=1024,
    num_heads=8,
    head_dim=256,
    num_kv_heads=1,
).to(device=device, dtype=dtype)


x = torch.zeros([batch_size, seq_length, 1024], device=device, dtype=dtype)

attn_mask = torch.bernoulli(
    torch.zeros([batch_size, 1, seq_length, seq_length], dtype=dtype) + 0.5,
    generator=torch.Generator().manual_seed(42)
) * torch.finfo(dtype).min
attn_mask = attn_mask.to(device)

# Compile flex
attn(query=x, key=x, value=x, attn_mask=attn_mask, flex=True)


def time_attn(x, kwargs, depth, num_trials) -> list[float]:
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(num_trials):
        y = x
        start.record()
        for _ in range(depth):
            y, _ = attn(query=y, key=y, value=y, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return times

with torch.autocast('cuda', dtype=torch.bfloat16):
    times_sdpa = time_attn(x, {'attn_mask': attn_mask, 'flex': False}, depth, num_trials)
    times_flex = time_attn(x, {'attn_mask': attn_mask, 'flex': True}, depth, num_trials)

print("Time SDPA".center(80, "-"))
print(np.mean(times_sdpa))
print("Time Flex".center(80, "-"))
print(np.mean(times_flex))