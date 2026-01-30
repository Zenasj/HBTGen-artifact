import torch
from torch.optim import Adam, AdamW

def build_optim(optim, lr, beta1, beta2):
    param = torch.rand(2, 3, dtype=torch.float, device="cuda", requires_grad=True)
    param.grad = torch.rand_like(param)
    
    lr = torch.tensor(lr, device="cuda")
    betas = (torch.tensor(beta1, device="cuda"), torch.tensor(beta2, device="cuda"))
    print(optim.__name__, lr, betas)
    
    return optim([param], lr=lr, betas=betas)

from itertools import product

for optim, lr, beta1, beta2 in product([Adam, AdamW], [0.1, [0.1]], [0.9, [0.9]], [0.99, [0.99]]):
    optimizer = build_optim(optim, lr, beta1, beta2)
    try:
        optimizer.step()
        print(">>> No error occurred.")
    except Exception as e:
        print(">>>", e)