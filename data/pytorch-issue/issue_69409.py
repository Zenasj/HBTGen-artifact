import torch
tensor_1 = torch.randint(-1,1,torch.Size([3]), dtype=torch.int64)
tensor_2 = torch.rand([3], dtype=torch.float64)
import itertools
res2 = torch.tensor(list(itertools.product(tensor_1.tolist(), tensor_2.tolist())))
# succeed
res1 = torch.cartesian_prod(tensor_1, tensor_2)
# RuntimeError: meshgrid expects all tensors to have the same dtype