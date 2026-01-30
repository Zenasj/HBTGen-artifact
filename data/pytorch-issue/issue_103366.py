import torch
x1 = torch.as_tensor(0,dtype=torch.int32)
x2 = torch.as_tensor(0,dtype=torch.int32)
torch.inner(x1,x2)
## tensor(0, dtype=torch.int32)

x1 = torch.as_tensor([0],dtype=torch.int32)
x2 = torch.as_tensor([0],dtype=torch.int32)
torch.inner(x1,x2)
## RuntimeError: expected scalar type Long but found Int