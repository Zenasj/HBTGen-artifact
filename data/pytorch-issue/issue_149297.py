import torch
import torch.nn as nn

def test_pending_unbacked(self):
        class M(torch.nn.Module):
            @mark_compile_region
            def gn(self, x):
                u = x[0].item()
                return x * u

            def forward(self, x):
                for _ in range(4):
                    x = self.gn(x)
                return x
        
        torch._dynamo.config.capture_scalar_outputs = True
        torch.compile(M())(torch.randn(8))