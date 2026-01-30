import torch

a = [234678.5462495405945]
b = torch.tensor(a)
print(b.item())

a = [2.456778865432]  
b = torch.tensor(a)
b.item()  
2.4567787647247314

b = torch.tensor(a, dtype=torch.double)  
b.item()  
2.456778865432