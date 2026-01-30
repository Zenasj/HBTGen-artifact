import torch.nn as nn

import torch
arg_1 = [2,1,1]
arg_2 = [0.5,0.5]
arg_class = torch.nn.FractionalMaxPool2d(kernel_size=arg_1,output_ratio=arg_2,)
arg_3_0_tensor = torch.rand([20, 16, 50, 32], dtype=torch.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
print(res)
# res: RuntimeError: fractional_max_pool2d: kernel_size must either be a single Int or tuple of Ints

import torch
arg_1 = [2,1]
arg_2 = [0.5,0.5,0.6]
arg_class = torch.nn.FractionalMaxPool2d(kernel_size=arg_1,output_ratio=arg_2,)
arg_3_0_tensor = torch.rand([20, 16, 50, 32], dtype=torch.float32)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
res = arg_class(*arg_3)
print(res)
# res: tensor([[[[0.7426, 0.9191, 0.1807,  ..., 0.4967, 0.9169, 0.8869],...