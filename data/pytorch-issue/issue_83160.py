import torch.nn as nn

import torch
results={}
arg_1 = torch.rand([2], dtype=torch.float64)
arg_2 = 2
arg_3 = 0
results['res'] = torch.nn.functional.rrelu_(arg_1,arg_2,arg_3,)

import torch
results={}
arg_1= torch.rand([2], dtype=torch.float32)
arg_2 = 0.3
arg_3 = 0.1
arg_4 = True
arg_5 = False
results['res'] = torch.nn.functional.rrelu(arg_1,arg_2,arg_3,arg_4,arg_5,)