import torch

a = torch.zeros(3, 4, dtype = torch.int32)
b = torch.zeros(3, 4, dtype = torch.int32)

torch.bitwise_and(a, 1, out = b) #works

a.bitwise_and(1, out = b)
#TypeError: bitwise_and() received an invalid combination of arguments - got (int, out=Tensor), but expected one of:
# * (Tensor other)
# * (Number other)