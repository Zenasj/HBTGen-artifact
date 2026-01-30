import torch

def fn(x):
    y = torch.relu(x).sum()
    y.backward()
    return y

opt_fn = torch.compile(fn)
x = torch.ones(3, 3, device="cuda", requires_grad=True)
y = opt_fn(x)