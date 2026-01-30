import torch

In [13]: a=torch.randn(4)

In [14]: a.bitwise_right_shift(2) #there's nothing bitwise about this shift
Out[14]: tensor([-0.0470, -0.0847,  0.0964,  0.0141])

In [15]: a.bitwise_right_shift(1.5) #what does this even mean?
Out[15]: tensor([-0.0664, -0.1198,  0.1363,  0.0200])