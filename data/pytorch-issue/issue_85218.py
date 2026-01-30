import torch

tensor_0 = torch.full((5, 0,), 1, dtype=torch.float32, requires_grad=False)
tensor_1 = torch.full((5,), 1, dtype=torch.float32, requires_grad=False)
tensor_2 = torch.full((5, 5,), 1, dtype=torch.float32, requires_grad=False)
bool_3 = True
bool_4 = True
torch.ormqr(tensor_0, tensor_1, tensor_2, bool_3, bool_4)