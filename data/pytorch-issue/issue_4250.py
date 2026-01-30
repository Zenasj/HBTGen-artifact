import torch

def test_inplace_copy(self):
        x = Variable(torch.randn(4, 4), requires_grad=True)
        def f(x):
            out = Variable(torch.zeros(x.size()))
            out.copy_(x)
            return out
        trace, z = torch.jit.trace(f, (x, ), nderivs=0)
        torch._C._jit_pass_lint(trace)
        torch._C._jit_pass_dce(trace)
        self.assertExpectedTrace(trace)