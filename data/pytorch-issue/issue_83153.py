import torch.nn as nn

import torch
results={}
arg_1 = torch.rand([80, 192, 9, 9], dtype=torch.float32)
arg_2 = 6.0
arg_3 = 0.0
arg_4 = True
results['res'] = torch.nn.functional.hardtanh(arg_1,arg_2,arg_3,arg_4,)