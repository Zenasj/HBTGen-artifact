import torch.nn as nn

import torch

class StackTest(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dq, dk, dv):
        ctx.save_for_backward(dq, dk, dv)

        # dqkv = torch.stack([dq, dk, dv], dim=1)
        return dq

    @staticmethod
    def backward(ctx, dout):
        dq, dk, dv, = ctx.saved_tensors

        dqkv = torch.stack([dq, dk, dv], dim=1)
        return dq, None, None

class StackModel(torch.nn.Module):
    def __init__(self):
        super(StackModel, self).__init__()

    def forward(self, dq, dk, dv):
        ctx = StackTest.apply(dq, dk, dv)
        return ctx

bs = 10
dq = torch.randn([bs, 16, 64], dtype=torch.float16, requires_grad=True).cuda()
dk = torch.randn([bs, 16, 64], dtype=torch.float16, requires_grad=True).cuda()
dv = torch.randn([bs, 16, 64], dtype=torch.float16, requires_grad=True).cuda()

def call():
    model = StackModel()
    model.to('cuda:0')
    res = model(dq, dk, dv)
    res.sum().backward()

# call()
fn = torch.compile(call, backend='inductor', dynamic=False)
fn()