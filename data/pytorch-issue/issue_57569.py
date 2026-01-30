import torch
import torch.nn as nn

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("twos", 2.0 * torch.ones(3))
    def forward(self, x):
        return self.twos * x

m = M().eval()
m(torch.ones(3))
ms = torch.jit.script(m)
ms = torch.jit.freeze(ms)
ms = ms.to("cuda")
ms(torch.ones(3))  # => tensor([2., 2., 2.])
ms(torch.ones(3).cuda())

m = M().eval()
m(torch.ones(3))
ms = torch.jit.script(m)
ms = ms.to("cuda")
ms(torch.ones(3).cuda())  # => tensor([2., 2., 2.], device='cuda:0')

m = M().eval()
m(torch.ones(3))
ms = torch.jit.script(m)
ms = ms.to("cuda")
ms = torch.jit.freeze(ms)
ms(torch.ones(3).cuda())  # => tensor([2., 2., 2.], device='cuda:0')

m = M().eval()
ms = torch.jit.script(m)
ms = torch.jit.freeze(ms)
torch.jit.save(ms, "tmp.pth")
ms = torch.jit.load("tmp.pth", map_location="cuda")
ms(torch.ones(3).cuda())  # => tensor([2., 2., 2.], device='cuda:0')