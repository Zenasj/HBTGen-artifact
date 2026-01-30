import torch

timeline = []
class Log(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, name):
        ctx.name = name
        timeline.append('%s:forward' % name)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        name = ctx.name
        timeline.append('%s:backward' % name)
        return grad_output, None

a = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

a = Log.apply(a, 'a')
b = checkpoint(lambda b: Log.apply(b, 'b'), b)
out = torch.cat((a, b)).sum()

#                 +--> Log[a] --> a
# Sum --> Cat[a, b]
#                 +--> Checkpoint(Log[b]) --> b
out.backward()

assert timeline == \
    ['a:forward', 'b:forward', 'b:forward', 'b:backward', 'a:backward']
#    |----------------------|  |-----------------------|  |----------|
#          forward pass            Checkpoint(Log[b])        Log[a]