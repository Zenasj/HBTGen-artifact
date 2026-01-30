import torch.nn as nn

import torch
results={}
arg_1_0 = 576460752303423488
arg_1_1 = -274877906944
arg_1 = [arg_1_0,arg_1_1,]
arg_class = torch.nn.AdaptiveAvgPool2d(arg_1,)
arg_2_0 = torch.rand([16, 960, 4, 4], dtype=torch.float32)
arg_2 = [arg_2_0,]
results['res'] = arg_class(*arg_2)