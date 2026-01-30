import torch
t1 = torch.ones((10000,10000), dtype=torch.float32, device='cpu') * 1.01
t2 = torch.tensor([[1.01]], dtype=torch.float32, device='cpu').expand((10000,10000))
t3 = torch.tensor([[1.01]], dtype=torch.float32, device='cpu').repeat((10000,10000))
print(t1.sum(), t1.mean())
print(t2.sum(), t2.mean())
print(t3.sum(), t3.mean())