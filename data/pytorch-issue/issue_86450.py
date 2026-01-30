import torch

tensor = torch.tensor([], dtype=torch.float32, device=device)
our_ctypes_lib.some_function(tensor._cdata)