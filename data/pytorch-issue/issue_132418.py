import torch
import torch.nn as nn

def test_with_effects_no_grad(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                with torch.no_grad():
                    torch._print("moo")
                    z = x + y
                return x * z 
            
        ep = torch.export.export(M(), (torch.ones(3, 3), torch.ones(3, 3)))
        print(ep)
        inp = (torch.randn(3, 3), torch.randn(3, 3))
        m = ep.module()
        self.assertTrue(torch.allclose(M()(*inp), m(*inp)))