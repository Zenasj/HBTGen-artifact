import torch

torch._dynamo.config.cache_size_limit = 1000
flex_attention_compile= torch.compile(flex_attention, dynamic=True)
data_type = torch.float16

def scorer_test(score, b, h, q_idx, kv_idx):
    mask = (q_idx >= kv_idx)
    return torch.where(mask, score, -float("inf"))

device = 'cuda'
B, H, S, D = 288, 16, 20, 64
q = torch.randn(B, H, S, D, device="cuda", dtype=data_type, requires_grad=True)
k = torch.randn(B, H, S, D, device="cuda", dtype=data_type, requires_grad=True)
v = torch.randn(B, H, S, D, device="cuda", dtype=data_type, requires_grad=True)
flex_out = flex_attention_compile(q, k, v, score_mod=scorer_test, block_mask=None)