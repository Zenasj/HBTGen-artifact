import torch
import torch.nn as nn

def test_runtime_asserts(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                b = x.item() 
                torch._check_is_size(b)
                torch._check(b < y.shape[0])
                return y[:b]
        
        ep = torch.export.export(M(), (torch.tensor(4), torch.randn(10)), dynamic_shapes=(None, {0: Dim.DYNAMIC}), strict=True)
        print(ep.module()(torch.tensor(4), torch.ones(10)))
        # print(ep.module()(torch.tensor(4), torch.ones(3)))  # errors, expected

        torch._dynamo.config.capture_scalar_outputs = True
        print(torch.compile(M(), fullgraph=True)(torch.tensor(4), torch.ones(10)))  # fails w/ DDE