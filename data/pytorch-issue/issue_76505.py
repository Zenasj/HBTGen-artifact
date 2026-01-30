import torch

with torch.jit.fuser('fuser2'):
    x = torch.rand((2, 2)).cuda()
    y = torch.rand((2, 2)).cuda()
    def fn(x, y):
        return x.sin() + y.exp()
    fn_s = torch.jit.script(fn)
    fn_s(x, y)
    fn_s(x, y)
    fn_s(x, y)