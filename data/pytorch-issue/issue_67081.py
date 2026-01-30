import torch

torch.utils.dlpack.to_dlpack(torch.BoolTensor([False, True]))   # Bool type is not supported by dlpack