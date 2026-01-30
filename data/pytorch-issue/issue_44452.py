import torch

x = torch.tensor([-0.5, 0, 0.5], requires_grad=True)
x.norm(p=0.5)  # 2.0
x.norm(p=0.5).backward()
x.grad # tensor([-2.0000,     nan,  2.0000])

x = torch.tensor([-0.5, 0, 0.5], requires_grad=True)
(x.abs() + 1e-6).norm(p=0.5)  # 2.0028
(x.abs() + 1e-6).norm(p=0.5).backward()
x.grad # tensor([-2.0014,  0.0000,  2.0014])