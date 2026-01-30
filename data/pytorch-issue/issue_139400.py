import torch

device = "mps"

X = torch.as_tensor([1,2],device=device)
stacked_X = torch.stack([X/2, X/2], dim=-1) # wihtout this line there is no bug

empty_tensor = torch.empty(0, 2, 2, device=device)

mask = X > empty_tensor

print(mask)
print(torch.count_nonzero(mask))

print(mask)
print(torch.count_nonzero(mask.cpu()))