import torch
# n can be any odd number
t = torch.tensor([[1.0/n]*n]).half().cuda()
torch.multinomial(t, 1, True)

# RuntimeError: CUDA error: misaligned address