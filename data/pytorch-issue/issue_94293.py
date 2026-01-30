import torch
import torch.nn as nn

class A(torch.nn.Module):
    def __init__(self, feature=4.0):
        super().__init__()
        self.feature = feature

    def forward(self, x):
        return int(x.shape[-1] * self.feature // 3)
       

torchdynamo.config.dynamic_shapes = True
torchdynamo.config.specialize_int_float = False
gm, _ = torchdynamo.export(A(), torch.ones(6, 1), aten_graph=True, tracing_mode="symbolic")
print(gm.graph)
print(gm(torch.ones(6, 1)))