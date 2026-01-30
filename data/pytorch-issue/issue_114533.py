import torch
x=torch.rand(1000, 1000, 100, device='cuda')
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.memory_allocated(i), torch.cuda.memory_reserved(i))