import torch

def fn(a):
    return 1.0 - a  # 0.99 - a works
a = torch.tensor([0.0, 0.49, 0.8, 0.9, 0.95])
y = fn(a)
print(y)  # tensor([1.0000, 0.5100, 0.2000, 0.1000, 0.0500])

a = a.to("mps")
y = fn(a)
print(y)  # tensor([1., 1., 0., 0., 0.], device='mps:0')