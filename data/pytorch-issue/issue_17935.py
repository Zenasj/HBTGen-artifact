import torch

a = torch.rand(3, 3)
b = a[:, 1]
balmostcontig, bstrideorig = torch.magic_op(b)
c = (balmostcontig * 3).as_strided(bstrideorig) # at maximum, c would have the same amount of raw memory as b, but in case of 0 strides much less