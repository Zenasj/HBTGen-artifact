import torch.nn as nn

import torch
from torch import nn
import torchviz

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(2, 3)
        self.fc4 = nn.Linear(3, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc4(x)
        print(x.size(), x.grad_fn, x.requires_grad)
        x = x**2
        return x.sum()


inp = torch.rand(1, 2).double()

model = VAE()
model.double()

print("Tracing")
ge = torch.jit.trace(model, (inp,))
print("Tracing done")

def get_grad_sum(*params):
    with torch.no_grad():
        for p_val, p in zip(params, model.parameters()):
            p.copy_(p_val)
    out = ge(inp)

    print(ge.code)

    gs = torch.autograd.grad(out, model.parameters(), create_graph=True)

    s = 0
    for g in gs:
        s += g.sum()
        print("grad", g.size())
    torchviz.make_dot(s).view()

    return s

s = get_grad_sum(*list(model.parameters()))
print("FW/BW ok")
torch.autograd.grad(s, model.parameters()) # Fails here
print("RUNS OK")

torch.autograd.gradcheck(get_grad_sum, list(model.parameters()))
print("GRAD CHECK OK")