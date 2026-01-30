import torch
sorted_sequence = torch.stack([torch.arange(10) for i in range(3)])
values = torch.randint(10, (3, 3))
# result = torch.searchsorted(sorted_sequence, values, side='left', right=True) # Running Error and Expected Error
result = torch.searchsorted(sorted_sequence, values, side='right', right=False) # Running Well but Expected Error!