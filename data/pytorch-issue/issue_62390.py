import torch
import torch.nn as nn
from torch.autograd.profiler import profile, record_function


class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 100)
        self.record = False

    def forward (self, x):
        if self.record:
            with record_function("a"):
                return self.l1(x)
        return self.l1(x)


class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.Linear(100, 100)
        self.record = False

    def forward (self, x):
        if self.record:
            with record_function("b"):
                return self.l2(x)
        return self.l2(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = A()
        self.b = B()


    def forward(self, x):
        y = self.a(x)
        y = self.b(x)
        y = self.b(x)
        return y


device = "cuda:0"
x = torch.ones((100, 100), device=device)
model = Net().to(device)

y = model(x)


######## Expectation: all calls should return same number of aten::zeros and aten::empty

# this returns 2 aten::zeros calls and 8 aten::empty calls

with profile(use_cuda=True, record_shapes=True) as prof:
    with record_function("model"):
        y = model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

model.a.record = True
model.b.record = False

# this returns 3 aten::zeros calls and 7 aten::empty calls

with profile(use_cuda=True, record_shapes=True) as prof:
    with record_function("model"):
        y = model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

model.a.record = True
model.b.record = True

# this returns 4 aten::zeros calls and 11 aten::empty calls

with profile(use_cuda=True, record_shapes=True) as prof:
    with record_function("model"):
        y = model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))