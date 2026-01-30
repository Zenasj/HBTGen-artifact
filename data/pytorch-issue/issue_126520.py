import torch

q = torch.randn([7, 1, 4, 256], dtype=torch.bfloat16, device='cuda')
k = torch.randn([7, 51, 1, 256], dtype=torch.bfloat16, device='cuda')
v = torch.randn([7, 51, 1, 256], dtype=torch.bfloat16, device='cuda')