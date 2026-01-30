import torch.nn as nn

import torch

torch._dynamo.config.dynamic_shapes = True

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.my_param = torch.nn.Parameter(torch.randn([2, 3]))

    def forward(self, x):
        return x + self.my_param

m = MyModule()

torch._dynamo.export(m, torch.randn([2,3]), aten_graph=True, tracing_mode="symbolic")