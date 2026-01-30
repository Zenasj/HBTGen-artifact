import torch.nn as nn

py
import torch

torch.manual_seed(420)

x = torch.randn(1, 3)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(3, 4)
        self.fc2 = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = self.fc1(x)
        y = torch.mm(x, torch.ones((4, 2)))
        x = self.fc2(x)
        z = torch.mm(x, torch.ones((2, 1)))
        out = torch.add(y, z)
        return out

func = Model().to('cpu')

with torch.no_grad():
    func.train(False)
    res1 = func(x) # without jit
    print(res1)

    jit_func = torch.compile(func)
    res2 = jit_func(x)
    print(res2)
    # extern_kernels.addmm(buf4, buf5, buf6, alpha=1, beta=1, out=buf7)
    # RuntimeError: The expanded size of the tensor (1) must match the existing size (2) at non-singleton dimension 1.  Target sizes: [1, 1].  Tensor sizes: [1, 2]