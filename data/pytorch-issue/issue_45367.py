import torch

data = torch.ones(3, device=torch.device("cuda"))

data1 = torch.as_tensor(data)
print(data1.device)

data2 = torch.as_tensor(data, dtype=data.dtype)
print(data2.device)

cuda:0
cpu