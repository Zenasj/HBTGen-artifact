import torch
a = torch.ones(2,3,6).float()
b = torch.sum(a)
print(b)
# got tensor(28.), it should be 36
c = torch.mean(a)
print(c)
# tensor(0.7778), it should be 1.