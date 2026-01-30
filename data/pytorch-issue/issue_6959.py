import torch.nn as nn

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

num_inp = 5000

nz_inp = 10
nz_out = 10
nz_bottleneck = 1000

# small proxy network for some complex reasoning we want to do per input
module = nn.Sequential(
    nn.Linear(nz_inp, nz_bottleneck),
    nn.ReLU(),
    nn.Linear(nz_bottleneck, nz_inp)
)

feat_combined = []
for r in range(num_inp):
    data_r = torch.Tensor(1, nz_inp)
    data_r.uniform_()
    data_r.requires_grad=True
    # feat_r = module(data_r) # using this instead of checkpoint line below works fine
    feat_r = checkpoint(module, data_r)

    feat_combined.append(feat_r)

# compute mean as a proxy for some joint reasoning
mean_combined = torch.stack(feat_combined).mean()

# the backward pass makes the code crash
mean_combined.backward()

a = checkpoint(model, a)
b = checkpoint(model, b)
c = torch.cat((a, b))
#                a   b
#                |   |
# checkpoint{model} checkpoint{model}
#                 \ /
#                 cat
#                  |
#                  c
loss = c.mean()
loss.backward()

timeline = []
class LogBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, tag):
        ctx.tag = tag
        return input
    @staticmethod
    def backward(ctx, grad_output):
        timeline.append(ctx.tag)
        return grad_output, None

a = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)
model = torch.nn.Linear(1, 1)

a = LogBackward.apply(a, 'a:end')
a = checkpoint(model, a)
a = LogBackward.apply(a, 'a:begin')

b = LogBackward.apply(b, 'b:end')
b = checkpoint(model, b)
b = LogBackward.apply(b, 'b:begin')

c = torch.cat((a, b))
loss = c.mean()
loss.backward()