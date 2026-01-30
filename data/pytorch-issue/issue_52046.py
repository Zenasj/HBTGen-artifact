import torch
a = torch.randn(3, 3)
# torch.linalg.inv has two different LAPACK calls, hence two Tensors for error codes
info1 = torch.empty(1)
info2 = torch.empty(1)
a_inverse = torch.linalg.inv(a, infos=(info1, info2))

# or for batched input case
a = torch.randn(5, 3, 3)
info1 = torch.empty(5)
info1 = torch.empty(5)
a_inverse = torch.linalg.inv(a, infos=(info1, info2))

a_inverse, info1, info2 = torch.linalg.inv(a, infos=True)