import torch

tensor = torch.utils.dlpack.from_dlpack(sampl.toDlpack())
print(tensor.dtype)  # --> torch.float64