import torch.nn as nn

import torch
import torchdynamo


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 4)

    def forward(self, x):
        return self.embedding(x)


m = Module()

torchdynamo.config.capture_scalar_outputs = True
torchdynamo.config.guard_nn_modules = True
torchdynamo.config.dynamic_shapes = True
torchdynamo.config.specialize_int_float = True
torchdynamo.config.verbose = True
torchdynamo.config.fake_tensor_propagation = False
torchdynamo.reset()

gm, _ = torchdynamo.export(m, torch.ones(10, dtype=torch.int64), aten_graph=True, tracing_mode="symbolic")
print(gm.graph)