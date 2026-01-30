import torch.nn as nn

import torch

arg_1_tensor = torch.rand([20, 16, 50], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([33, 16, 3], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.rand([33], dtype=torch.float32)
arg_3 = arg_3_tensor.clone()
arg_4_0 = 2**32
arg_4 = [arg_4_0]
arg_5_0 = 0
arg_5 = [arg_5_0]
arg_6_0 = 1
arg_6 = [arg_6_0]
arg_7 = 1
res = torch.nn.functional.conv1d(arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7)