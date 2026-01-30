import torch
import torch.nn as nn
from torch.autograd import grad

print(torch.__version__)

imsv = torch.rand(1, 3, 6, 6, requires_grad=True)

x_s1_2 = nn.PReLU(num_parameters=3)(imsv)

tmp = nn.ZeroPad2d([0, 1, 0, 1])(x_s1_2)
preds = nn.AvgPool2d(kernel_size=2,
                        stride=1,
                        padding=0,
                        ceil_mode=True,
                        count_include_pad=False)(tmp)

g, = grad(preds.sum(), imsv, create_graph=True)

grad(g.sum(), imsv)