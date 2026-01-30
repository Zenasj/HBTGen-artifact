import torch
x = torch.Size([1,2,3])
y = torch.Size([4,5,6])
reveal_type(x+y)  # tuple[int, ...], not Size !!!