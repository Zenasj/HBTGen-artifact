import torch.nn as nn

import torch
results={}
arg_1 = 2
arg_2 = 50
arg_3 = True
arg_class = torch.nn.MaxPool1d(arg_1,padding=-1,stride=arg_2,return_indices=arg_3,)
arg_4 = torch.rand([0, 1, 49], dtype=torch.float32)
results['res'] = arg_class(arg_4)

import torch
results={}
arg_1_tensor = torch.rand([1, 1, 9], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 2
arg_3 = 2
arg_4 = -1
arg_5 = 1
arg_6 = False
arg_7 = False
results['res'] = torch.nn.functional.max_pool1d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)