import torch
def bar():
    def test(a):
        return a
    x = torch.ones(10,10, device='cpu')
    print(torch.jit.trace(test, (x)).graph)
bar()