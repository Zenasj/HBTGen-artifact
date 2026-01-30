import torch

torch.FloatTensor([0.0, 1.0]) > 0  # works
# tensor([0, 1], dtype=torch.uint8)

torch.FloatTensor([0.0, 1.0]) > torch.tensor(0) # fails
# RuntimeError: Expected object of scalar type Float but got scalar type Long for argument #2 'other'