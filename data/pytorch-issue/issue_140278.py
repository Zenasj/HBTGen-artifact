py
import torch

q = torch.randn([16, 16, 1024, 64], dtype=torch.float16, device='cuda')
k = torch.randn([16, 16, 1024, 64], dtype=torch.float16, device='cuda')

q_chunks = q.split(512, dim = -2)
k_chunks = k.split(1024, dim = -2)

# Memory access fault on ROCm!
torch.einsum('b h i d, b h j d -> b h i j', q_chunks[0], k_chunks[0])

import torch

q = torch.randn([16, 16, 1024], dtype=torch.float16, device='cuda')
k = torch.randn([16, 16, 1024], dtype=torch.float16, device='cuda')

q_view = q.view(256, 1024)
k_view = k.view(256, 1024)

C = torch.matmul(q_view, k_view)

import torch

i = 0
j = 0

q = torch.randn([16, 16, 1024, 64], dtype=torch.float16, device='cuda')
k = torch.randn([16, 16, 1024, 64], dtype=torch.float16, device='cuda')

q_chunks = q.split(512, dim = -2)
k_chunks = k.split(64, dim = -2)

print((q_chunks[i]).size())
print((k_chunks[j]).size())

C = torch.matmul(q_chunks[i], k_chunks[j]) # seg fault with TunableOp
# C = torch.matmul(q_chunks[i].contiguous(), k_chunks[j].contiguous()) # OK with TunableOp