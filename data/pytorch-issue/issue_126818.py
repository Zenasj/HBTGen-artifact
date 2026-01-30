import torch

a = torch.tensor([1.0], device="cuda")
b = torch.tensor([2.0], device="cuda")
pred = torch.tensor(True, device="cuda")
c = torch.cond(pred, lambda x, y: x + y, lambda x, y: x * y, [a, b]) # runs correctly 
c = torch.cond(pred, torch.add, torch.mul, [a, b]) # crashes