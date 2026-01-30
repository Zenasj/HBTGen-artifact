import torch.nn as nn

import torch

def test():
    arg_1_0_tensor = torch.rand([1024, 255, 255], dtype=torch.float32)
    arg_1_0 = arg_1_0_tensor.clone()
    arg_1_1_tensor = torch.rand([1, 255], dtype=torch.float16)
    arg_1_1 = arg_1_1_tensor.clone()
    arg_1_2_tensor = torch.rand([1, 0, 0], dtype=torch.float32)
    arg_1_2 = arg_1_2_tensor.clone()
    arg_1_3_tensor = torch.rand([1, 255, 0], dtype=torch.float32)
    arg_1_3 = arg_1_3_tensor.clone()
    arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
    arg_2 = 0
    res = torch.nn.utils.rnn.pack_sequence(arg_1,arg_2,)

test()