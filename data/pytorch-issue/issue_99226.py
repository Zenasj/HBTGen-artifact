import torch.nn as nn

import torch
arg_1 = 1024
arg_2 = 64
arg_3 = False
arg_class = torch.nn.Linear(in_features=arg_1,out_features=arg_2,bias=arg_3,)
arg_4_0_tensor = torch.randint(-4096,1,[11, 0, 1024], dtype=torch.int16)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
res = arg_class(*arg_4)
print(res)