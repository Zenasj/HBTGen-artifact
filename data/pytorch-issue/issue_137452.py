import torch

bias = torch.randn(
    params.seq_length,
    device=self.device,
    dtype=params.dtype,
    requires_grad=True,
)

offset = torch.randint(
    0,
    params.seq_length,
    (params.seq_length,),
    device=self.device,
)

def score_mod(score, b, h, q_idx, kv_idx):
    return score + bias[offset[q_idx]]