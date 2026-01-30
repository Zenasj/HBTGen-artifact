import torch

@torch. jit.script
class MyFunc(object):
    def caller(self, x):
        return x + 100


@torch.jit.script
class MyFunc2(MyFunc):
    def caller(self, x):
        return x * 99


@torch.jit.script
def other(x, fn):
    # type: (Tensor, MyFunc)
    return fn.caller(x)

print(other(torch.ones(2, 2), MyFunc2()))