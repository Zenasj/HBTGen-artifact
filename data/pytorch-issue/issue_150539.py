import torch
import torch.nn as nn

def test_aoti_runtime_asserts(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                b = x.item() 
                torch._check_is_size(b)
                torch._check(b < y.shape[0])
                return y[:b]
        
        ep = torch.export.export(M(), (torch.tensor(4), torch.randn(10)), dynamic_shapes=(None, {0: Dim.DYNAMIC}), strict=True)
        print(ep)
        path = torch._inductor.aoti_compile_and_package(ep)
        compiled_m = torch._inductor.aoti_load_package(path)
        print(compiled_m(torch.tensor(4), torch.ones(10)))
        print(compiled_m(torch.tensor(4), torch.ones(3)))  # should error, but instead returns tensor([1, 1, 1, 0])

        print(M()(torch.tensor(4), torch.ones(10))) 
        print(M()(torch.tensor(4), torch.ones(3)))  # errors