import torch
arg_1 = (2, 2)
arg_2 = [25,40]
arg_3 = 1
results = torch.as_strided(size=arg_1,stride=arg_2,storage_offset=arg_3,)