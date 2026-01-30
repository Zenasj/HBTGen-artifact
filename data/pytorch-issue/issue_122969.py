import torch

m = torch.compile(m)
m(*inputs)
del m, inputs
gc.collect()
# without compile, all the tensors from m are gone from memory
# with compile, some tensors remain