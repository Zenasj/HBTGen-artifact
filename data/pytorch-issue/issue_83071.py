import torch.nn as nn

import torch
results={}
arg_1 = torch.rand([1, 64, 8], dtype=torch.float32)
arg_2 = -255
arg_3 = False
results['res'] = torch.nn.functional.adaptive_max_pool1d(arg_1,arg_2,arg_3,)
print(results['res'].shape)
#torch.Size([1, 64, -255])

import torch
results={}
arg_1 = torch.rand([1, 64, 8, 9], dtype=torch.float32)
arg_2 = [17,-18]
arg_3 = False
results['res'] = torch.nn.functional.adaptive_max_pool2d(arg_1,arg_2,arg_3,)
print(results['res'].shape)
#torch.Size([1, 64, 17, -18])

import torch
results={}
arg_1= torch.rand([1, 64, 8, 9, 10], dtype=torch.float32)
arg_2 = [5,-7,9]
arg_3 = False
results['res'] = torch.nn.functional.adaptive_max_pool3d(arg_1,arg_2,arg_3,)
print(results['res'].shape)
#torch.Size([1, 64, 5, -7, 9])