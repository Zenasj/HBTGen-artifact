import torch

def fn(x, y):
    x = torch.sum(x.view(int(x.shape[0]/6), 6), dim=1)
    return torch.gather(x, 0, torch.trunc(y).to(torch.int64))
    
x = torch.randn(30, device='cuda')
y = torch.ones(1, dtype=torch.float64, device='cuda')
torch.compile(fn)(x, y)
x = torch.randn(36, device='cuda')
torch.compile(fn)(x, y)