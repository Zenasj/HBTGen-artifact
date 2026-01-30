import torch
foo = torch.arange(5)
foo.as_strided((5,), (-1,), storage_offset=4)