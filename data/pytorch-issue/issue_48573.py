dummy = torch.zeros(1, 2048, 2048, 4, device='cuda:0')
torch.mean(torch.zeros_like(dummy)[..., :3])

dummy = torch.zeros(1, 2048, 2048, 3, device='cuda:0')
torch.mean(torch.zeros_like(dummy)[..., :3])

dummy = torch.zeros(3, device='cuda:0')
dummy.expand(1, 2048, 2048, 3).mean()

dummy = torch.zeros(3, device='cuda:0')
dummy.expand(1, 2048, 32, 3).mean()

import torch
dummy = torch.zeros(1, 2048, 2048, 4, device='cuda:0')
torch.mean(torch.zeros_like(dummy)[..., :3])

ptrhon
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)