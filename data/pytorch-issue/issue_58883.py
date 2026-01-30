import torch.nn as nn

import torch
i = [[0, 1, 1],
      [2, 0, 2]]
v =  [3, 4, 5]
s = torch.sparse_coo_tensor(i, v, (2, 3))

class M(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
        self.register_buffer("x", x)
    def forward(self, y):
        return torch.mm(self.x, y)

m = M(s)
torch.save(m, "tmp.pth")

ms = torch.jit.script(m)
# => [W pybind_utils.cpp:31] Warning: Using sparse tensors in TorchScript is experimental. Many optimization pathways have not been thoroughly tested with sparse tensors. Please include the fact that the network is running sparse tensors in any bug reports submitted. (function operator())
torch.jit.save(ms, "tmp.pth")

class M(torch.nn.Module):
     def __init__(self, xind, xval):
         super().__init__()
         self.register_buffer("xind", xind)
         self.register_buffer("xval", xval)
     def forward(self, y):
         x = torch.sparse_coo_tensor(self.xind, self.xval, (2, 3))
         return torch.mm(x, y)

m = M(torch.LongTensor(i), torch.Tensor(v))
m(torch.randn(3, 2)).shape  # => torch.Size([2, 2])

ms = torch.jit.script(m)
ms(torch.randn(3, 2)).shape  # => torch.Size([2, 2])
torch.jit.save(ms, "tmp.pth")
ms2 = torch.jit.load("tmp.pth")
ms2(torch.randn(3, 2)).shape  # => torch.Size([2, 2])