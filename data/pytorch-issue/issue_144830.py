import torch

@torch.compile
def to_binary_string(num: int):
    return format(num, "b")

to_binary_string(10)

import torch

@torch.compile
def to_binary_string(num: int):
     "b".format(num)

to_binary_string(10)