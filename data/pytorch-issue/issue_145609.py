import torch
randint_result = torch.randint(0, 10, (100, 100), requires_grad=True) # failed
# randint_like_result = torch.randint_like(torch.randint(0, 10, (100, 100)), 0, 10, requires_grad=True) # failed
# randperm_result = torch.randperm(10, requires_grad=True) # failed