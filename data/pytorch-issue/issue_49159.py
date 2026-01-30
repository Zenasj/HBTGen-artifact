import torch.nn as nn

import torch
x = torch.tensor([[[[2., 4.],[6., 4.]],[[5., 7.], [5., 2.]]], [[[9., 1.],[6., 7.]],[[9., 2.],[9., 4.]]]])
y = torch.nn.functional.interpolate(x, mode = 'bicubic', scale_factor =1)
print(y)

tensor([[[[2., 4.],[6., 4.]],[[5., 7.],[5., 2.]]],[[[0., 0.],[0., 0.]], [[0., 0.],[0., 0.]]]])

tensor([[[[2., 4.], [6., 4.]],[[5., 7.], [5., 2.]]],[[[9., 1.],[6., 7.]],[[9., 2.],[9., 4.]]]], device='cuda:0')