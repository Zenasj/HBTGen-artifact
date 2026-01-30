import torch

def test():
    arg_1 = torch.rand([], dtype=torch.float32).clone()
    arg_3 = torch.zeros([2], dtype=torch.int64).clone()
    res = torch.Tensor.index_select(arg_1,0,arg_3,)

test()