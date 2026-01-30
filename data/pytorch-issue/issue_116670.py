import torch.nn as nn

import torch

class InnerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner_module = InnerModule()

    def forward(self, x, y):
        out = x + y
        out = self.inner_module(out)
        out = out + x
        out = self.inner_module.relu(out)
        return out

exported_program = torch.export.export(
    TestModule(), args=(torch.randn(3), torch.randn(3))
)

for node in exported_program.graph.nodes:
    if node.op != "output":
        try:
            print(node.meta["val"])
        except KeyError:
            print("ExportedProgram lacks of val in meta")

unflattened_module = torch.export.unflatten(exported_program)
for node in unflattened_module.graph.nodes:
    if node.op != "output":
        try:
            print(node.meta["val"])
        except KeyError:
            print("UnflattenedModule lacks of val in meta")