import torch.nn as nn

from torch import nn
import torch
from torch.autograd import Function
torch.ops.load_library("op/build/lib.linux-x86_64-cpython-311/op.so")
class CalculateForce(Function):
    @staticmethod
    def forward(ctx, bb, aa, x):
        ctx.save_for_backward(bb)
        print('calculate foward 0')
        #torch.ops.op.calculate_force_forward(bb,aa,x)
        print('calculate foward 1')
        return x
    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        bb = inputs[0]
        grad = torch.zeros_like(bb)
        print('grad_output',grad_output.shape)
        #torch.ops.op.calculate_force_backward(bb, grad_output, grad)
        return (None, grad, None)
class SANNet(nn.Module):
    def __init__(self):
        super(SANNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_features=200000, out_features=1, bias=True))
        self.net1 = nn.Parameter(torch.Tensor(1,200000))
    def forward(self, feat):
        feat.requires_grad_()
        aa = torch.randn(1,200000)
        m = torch.ones(1,200000)
        m = feat * aa
        x = torch.ones(1,200000)
        x = CalculateForce.apply(feat, aa, x)
        return aa
model = SANNet()
print(model)
x = torch.ones(1,200000)
y = model(x)
ms = torch.jit.script(model)
ms.save('aa.script')