import torch
device = torch.device('cuda')
generator = torch.manual_seed(123)
x = torch.zeros(10, device=device)
x.uniform_(-1.0, 1.0, generator=generator)

device = torch.device('cuda')
g1 = torch.random_seed(123)
if condition:
    g2 = torch.random_seed(456)
    y = torch.empty(10, device=device)
    y.uniform_(generator=g2)
x = torch.empty(10, device=device)
x.uniform_(generator=g1)