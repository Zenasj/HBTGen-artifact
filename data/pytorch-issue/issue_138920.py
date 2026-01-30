import torch.nn as nn

import torch

class Mod(torch.nn.Module):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = a
        self.c = c
        self.b = b
        self.lin1 = torch.nn.Linear(b * a, b * c, device='cuda')

    def forward(self, x):
        x = x.view(-1, self.a * self.b)
        y = self.lin1(x)
        y = y.view(-1, self.c, self.b).contiguous()
        y = torch.flatten(y, start_dim=1)
        return y

class Mod2(torch.nn.Module):
    def __init__(self, a, b, c):
        super().__init__()
        self.mod = Mod(a, b, c)

    def forward(self, s, tensor_dict):
        args = tensor_dict[s]
        x = torch.cat(list(args))
        out = self.mod(x)
        return out

class Mod3(torch.nn.Module):
    def __init__(self, mods):
        super().__init__()
        self.mods = mods

    def forward(self, strs, tensor_dict, x):
        outs = [x]
        for i, m in enumerate(self.mods):
            s = strs[i]
            print("graph break")
            out = m(s, tensor_dict)
            outs.append(out)
        return torch.cat(outs).sum(0)

mods = [
    Mod2(192, 1, 48),
    Mod2(336, 1, 48),
]
m = Mod3(mods)

strs = ['a', 'b']

def gen_tensor_dict(sizes):
    tensor_dict = {
        'a': [
            torch.randn(sizes[0], 48, device='cuda') for _ in range(4)
        ],
        'b': [
            torch.randn(sizes[1], 48, device='cuda') for _ in range(7)
        ],
    }
    return tensor_dict


m = torch.compile(m)

with torch._dynamo.utils.maybe_enable_compiled_autograd(
    True, fullgraph=True, dynamic=False
):
    x = torch.zeros(100, 48, device='cuda')
    tensor_dict = gen_tensor_dict([101, 102])

    out = m(strs, tensor_dict, x)
    out.sum().backward()

    x = torch.zeros(103, 48, device='cuda')
    tensor_dict = gen_tensor_dict([104, 105])

    out = m(strs, tensor_dict, x)
    out.sum().backward()