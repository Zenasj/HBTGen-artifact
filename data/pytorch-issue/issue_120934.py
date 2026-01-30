import torch

scheduler = LambdaLR(optimizer, lr_scheduling_func)

lr_scheduling_func = lambda iter: torch.tensor(lr_scheduling_func(iter), requires_grad=False)
scheduler = LambdaLR(optimizer, lr_scheduling_func)