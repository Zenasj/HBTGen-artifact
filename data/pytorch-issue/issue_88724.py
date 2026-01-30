import torch.nn as nn

import torch

def test():
    arg_class = torch.nn.MultiMarginLoss()
    arg_1_0_tensor = torch.rand([3, 5], dtype=torch.float32)
    arg_1_0 = arg_1_0_tensor.clone().cuda()
    arg_1_1_tensor = torch.randint(-1,0,[3], dtype=torch.int64)
    arg_1_1 = arg_1_1_tensor.clone().cuda()
    arg_1 = [arg_1_0,arg_1_1,]
    res = arg_class(*arg_1)

test()