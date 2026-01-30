import torch
x = torch.randn(3, requires_grad=True)
def f(inp):
    return (x + inp).sum()

tf = torch.jit.trace(f, (torch.randn(1),))
f(torch.randn(1))
# expected output: tensor(-1.2899, grad_fn=<SumBackward0>) result requires grad

tf(torch.randn(1))
# got output: tensor(-3.6019), result does not require grad!