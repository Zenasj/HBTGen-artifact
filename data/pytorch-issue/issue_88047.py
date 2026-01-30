import torch.nn as nn

import torch

def test():
    ctc_loss = torch.nn.CTCLoss()
    arg_1_0 = torch.rand([50, 16, 20], dtype=torch.float32).clone()
    arg_1_1 = torch.randint(-512,4,[16, 30], dtype=torch.int64).clone()
    arg_1_2 = torch.randint(-4,0,[16], dtype=torch.int64).clone()
    arg_1_3 = torch.randint(-512,0,[16], dtype=torch.int64).clone()
    arg_1 = [arg_1_0,arg_1_1,arg_1_2,arg_1_3,]
    res = ctc_loss(*arg_1)

test()