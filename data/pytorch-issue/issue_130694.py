import torch

tensor1 = torch.tensor(2)
tensor2 = torch.tensor(7)
tensor3 = torch.tensor(4)

torch.column_stack(tensor1, tensor2, tensor3) # Error

torch.column_stack((tensor1, tensor2, tensor3))
torch.column_stack([tensor1, tensor2, tensor3])
torch.column_stack(tensors=(tensor1, tensor2, tensor3))
# tensor([[2, 7, 4]])