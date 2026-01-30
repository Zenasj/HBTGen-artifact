import torch.nn as nn

import torch
results={}
arg_1 = -7
arg_class = torch.nn.AdaptiveMaxPool3d(arg_1,)
arg_2_0_tensor = torch.rand([1, 64, 10, 9, 8], dtype=torch.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
results['res'] = arg_class(*arg_2)
print(results['res'].shape)
#torch.Size([1, 64, -7, -7, -7])

import torch
results={} 
arg_1 = -1
arg_class = torch.nn.AdaptiveMaxPool2d(arg_1,)
arg_2_0_tensor = torch.rand([1, 43, 8, 9], dtype=torch.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
results['res'] = arg_class(*arg_2)
print(results['res'].shape)
#torch.Size([1, 43, -1, -1])