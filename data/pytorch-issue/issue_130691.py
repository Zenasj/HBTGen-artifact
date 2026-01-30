import torch

tensor1 = torch.tensor([2, 7, 4])
tensor2 = torch.tensor([8, 3, 2])
tensor3 = torch.tensor([5, 0, 8])

torch.cat(tensor1, tensor2, tensor3) # Error

torch.cat((tensor1, tensor2, tensor3))
torch.cat([tensor1, tensor2, tensor3])
torch.cat(tensors=(tensor1, tensor2, tensor3))
# tensor([2, 7, 4, 8, 3, 2, 5, 0, 8])