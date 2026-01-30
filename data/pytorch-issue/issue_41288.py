import torch


MY_STR_CONST = "Hi, I am a string, please realize I am a constant"


@torch.jit.script
def fn():
    return MY_STR_CONST


print(fn())