import torch

t1 = torch.tensor([0, 1, 2])
t2 = torch.tensor([3, 4, 5])

torch.dot(input=t1, other=t2) # Error
t1.dot(other=t2) # Error

torch.dot(input=t1, tensor=t2) # tensor(14)
t1.dot(tensor=t2) # tensor(14)

torch.__version__ # 2.2.1+cu121