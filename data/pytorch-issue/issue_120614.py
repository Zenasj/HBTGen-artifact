import torch

torch_array = torch.tensor(other_array)

from torch.utils.dlpack import from_dlpack
torch_array = from_dlpack(other_array)