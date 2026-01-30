import torch.nn as nn

import torch




class M(torch.nn.Module):
    def forward(self, has_pads, x):
        y = torch.cond(
            has_pads,
            lambda: x + 1,
            lambda: x + 2,
            )
        
        z = torch.cond(
            has_pads,
            lambda: y + 1,
            lambda: y + 2,
            )
        return (z, )


ep = torch.export.export(M(), (torch.tensor([True]), torch.tensor([1, 2, 3])))
torch._inductor.aoti_compile_and_package(ep)