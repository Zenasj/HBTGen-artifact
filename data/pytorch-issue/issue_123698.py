import torch

values = torch.rand((8, 4))
offsets = torch.tensor([0, 2, 5, 6, 8])
nt = torch.nested.nested_tensor_from_jagged(values, offsets)
print(nt.shape)  # torch.Size([4, j2, 4])
print(nt._has_symbolic_sizes_strides)  # False