import torch
import numpy as np

def test_dynamic_backward(self):
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                x = torch.cat([x, x])
                return torch.addmm(x, x, x).relu(), x.size(0)

            inp = torch.randn(2, 4, device="cuda", requires_grad=True)
            r1, s1 = foo(inp)
            r1.sum().backward()
            g1 = inp.grad
            inp.grad = None
            r2, s2 = foo(inp)
            r2.sum().backward()
            g2 = inp.grad
            inp.grad = None
            self.assertEqual(r1, r2)
            self.assertEqual(s1, s2)
            self.assertEqual(g1, g2)