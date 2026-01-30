import torch
import logging

torch._logging.set_logs(dynamo=logging.DEBUG, inductor=logging.DEBUG)
torch._dynamo.config.suppress_errors = False


def fn(x1, x2, x3, w1, w2, w3):
    x = torch.stack([x1, x2, x3])
    w = torch.stack([w1, w2, w3])

    y = torch.bmm(x, w)

    return y

x1 = torch.randn(5, 4).cuda()
x2 = x1 + 1
x3 = x1 + 2
w1 = torch.randn(4, 3).cuda()
w2 = w1 + 1
w3 = w1 + 2

args = [x1, x2, x3, w1, w2, w3]

ref = fn(*args)
print(ref)

res = torch.compile(fn)(*args)
print(res)

buf3

buf7

mm

bmm

mm

bmm