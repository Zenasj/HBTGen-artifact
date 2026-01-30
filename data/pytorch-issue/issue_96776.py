import torch

a = torch.ones((), dtype=torch.float32, device='cuda')
b = torch.tensor([2], dtype=torch.float32, device='cuda')
print(a.size())  # torch.Size([])
print(b.size())  # torch.Size([1])
print(torch.div(a, b))  # OK: tensor([0.5000], device='cuda:0')
torch.div(a, b, out=a)  # FAILS: RuntimeError: output with shape [] doesn't match the broadcast shape [1]
torch._foreach_div_([a], [b])  # FAILS: RuntimeError: output with shape [] doesn't match the broadcast shape [1]