import torch


def torch_memory(device):
    # Checks and prints GPU memory
    print(f'{torch.cuda.memory_allocated(device)/1024/1024:.2f} MB USED')
    print(f'{torch.cuda.memory_reserved(device)/1024/1024:.2f} MB RESERVED')
    print(f'{torch.cuda.max_memory_allocated(device)/1024/1024:.2f} MB USED MAX')
    print(f'{torch.cuda.max_memory_reserved(device)/1024/1024:.2f} MB RESERVED MAX')
    print('')


device = torch.device(0)

a = torch.randn((1, 32, 24, 512, 512), dtype=torch.float32, device=device)  # 768 MB tensor
torch_memory(device)

indices = a < 0  # 192 MB tensor
torch_memory(device)

a[indices] = 0.  # Allocates 10 times of 768 MB
torch_memory(device)

a.index_put_((indices,), torch.tensor([0.]).to(device))

a[indices]