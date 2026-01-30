import torch

x = torch.rand(100, 30, 8, device='cpu')
a = torch.randn(8, 20, device='cpu')

batch_size = 2
y1 = torch.cat([torch.matmul(x[i:i+batch_size], a) for i in range(0, len(x), batch_size)])

batch_size = 50
y2 = torch.cat([torch.matmul(x[i:i+batch_size], a) for i in range(0, len(x), batch_size)])

print((y1 == y2).all())
(y1 - y2).pow(2).sum().pow(0.5)