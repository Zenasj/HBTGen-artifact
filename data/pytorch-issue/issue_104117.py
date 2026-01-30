import torch

from torch import _dynamo as dy

def func():
    a = torch.full((8, 8), 1)
    print(a.dtype)
    print(a.numpy().dtype)


func()
opt_func = dy.optimize('eager')(func)
opt_func()

torch.int64
int64
torch.float32
int64

from torch import _dynamo as dy

def func():
    return torch.full((8, 8), 1)


opt_func = dy.optimize('eager')(func)

# Correctly prints torch.int64, same as eager
print(opt_func().dtype)

torch.full