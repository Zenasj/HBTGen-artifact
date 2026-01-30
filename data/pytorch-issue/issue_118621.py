import torch.nn as nn

import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(100, 10)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))

model = MyModule().eval().cuda()
x = torch.randn(8, 100).cuda()
opt_model = torch.compile(model)
exp = torch.export.export(opt_model, (x,))
torch.export.save(exp, "model.ep")
new_model = torch.export.load("model.ep")
new_model(x)