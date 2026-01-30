import torch

# swap as it is expected
a,b = 0,1 
a,b = b,a 
assert a==1 and b==0

# 2 channel tensor, with zeros and ones
t = torch.stack((torch.zeros(8,8), torch.ones(8,8)))
assert t[0].sum().item() == 0 and t[1].sum().item() == 64

# Swap zeros in channel 0 with ones from channel 1
# This does not work as expected:
t[0,1::2,1::2], t[1,1::2,1::2] = t[1, 1::2, 1::2], t[0, 1::2, 1::2]

print(t)
assert t[0].sum().item() == 16
assert t[1].sum().item() == 48