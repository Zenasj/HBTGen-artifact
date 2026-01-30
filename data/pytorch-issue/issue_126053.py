import torch.nn as nn

import torch

with torch.autocast("cuda", enabled=True):
    print(f"{hidden_states.dtype=} {attn.to_q.weight.dtype=}", flush=True)
    # hidden_states.dtype=torch.float32, attn.to_q.weight.dtype=torch.float16
    
    query = to_q(hidden_states)
    print(f"{query.dtype=}", flush=True)
    # query.dtype=torch.float16

    key = to_k(hidden_states)
    value = to_v(hidden_states)

    print(f"{query.dtype=} {key.dtype=} {value.dtype=}", flush=True)
    # query.dtype=torch.float16 key.dtype=torch.float16 value.dtype=torch.float16

    query = attn.q_norm(query)
    key = attn.k_norm(key)
    print(f"{query.dtype=} {key.dtype=} {value.dtype=}", flush=True)
    # query.dtype=torch.float32 key.dtype=torch.float32 value.dtype=torch.float16

    hidden_states = xformers.ops.memory_efficient_attention(
        query, key, value
    )

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        print(f"{x.dtype=}", flush=True)
        # x.dtype=torch.float16
        with torch.autocast(device_type='cuda', enabled=False): # whether or not disable autocast locally does not affect issues
            output = self._norm(x.float()).type_as(x)
        print(f"{output.dtype=} {self.weight.dtype=}", flush=True)
        # output.dtype=torch.float16 self.weight.dtype=torch.float32
        output = output * self.weight
        print(f"{output.dtype=}", flush=True)
        # output.dtype=torch.float32
        return output