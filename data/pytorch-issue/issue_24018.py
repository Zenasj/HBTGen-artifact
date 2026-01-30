import torch

torch.zeros((16*2**20 - 512)//2 + 1, 1, dtype=torch.float16, device='cuda:0') @ torch.zeros(1, 2, dtype=torch.float16, device='cuda:0')

torch.zeros((16*2**20 - 512)//2, 1, dtype=torch.float16, device='cuda:0') @ torch.zeros(1, 2, dtype=torch.float16, device='cuda:0')