import torch

from flash_attn import flash_attn_func



q = torch.randn([1, 4096, 8, 128]).cuda().half()
k = torch.randn([1, 4096, 8, 128]).cuda().half()
v= torch.randn([1, 4096, 8, 128]).cuda().half()

result = flash_attn_func(q, k, v, causal=True)
print(result.shape)