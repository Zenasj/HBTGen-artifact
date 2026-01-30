import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        self.register_buffer("buf", torch.randn(4))

    def forward(self, x):
        x = self.fc(x)
        x = x + self.buf
        return x


m = Model()

for n, p in m.named_parameters():
    print("Before to(): parameter {} id={}".format(n, id(p)))
for n, b in m.named_buffers():
    print("Before to(): buffer {} id={}".format(n, id(b)))

m.to(torch.device("cuda"))

for n, p in m.named_parameters():
    print("After to(): parameter {} id={}".format(n, id(p)))
for n, b in m.named_buffers():
    print("After to(): buffer {} id={}".format(n, id(b)))