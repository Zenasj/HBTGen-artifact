import torch.nn as nn

import torch
def test():
    arg_1 = torch.rand([0, 16, 25, 16, 7], dtype=torch.float32).clone()
    arg_2 = torch.randint(-16,8,[20, 16, 25, 16, 7], dtype=torch.int64).clone()
    arg_3 = [42, 6]
    arg_4 = 2
    arg_5 = [0, 0, 0]
    arg_6 = None
    res = torch.nn.functional.max_unpool3d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,)

test()