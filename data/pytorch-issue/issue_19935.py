import torch.nn as nn

import torch
import onnx
from onnx import numpy_helper

class TestModule3(torch.nn.Module):
    def __init__(self):
        super(TestModule3, self).__init__()
        self.embeds = torch.nn.Embedding(2, 7)

    def forward(self):
        return None

class TestModule2(torch.nn.Module):
    def __init__(self, embed3):
        super(TestModule2, self).__init__()
        self.embeds = torch.nn.Embedding(2, 6)
        self.embed3 = embed3

    def forward(self):
        return None

class TestModule(torch.nn.Module):
    def __init__(self, embed2, embed3):
        super(TestModule, self).__init__()
        # When a member is initialized externally, the ONNX export
        # parameters orderings would be wrong.
        self.embeds = torch.nn.Embedding(2, 5)
        self.embeds2 = embed2
        self.embeds3 = embed3

    def forward(self, id1):
        return self.embeds(id1)

onnx_path = "external_member.onnx"
model3 = TestModule3()
model2 = TestModule2(model3)
model = TestModule(model2, model3)
torch_in = (torch.tensor(0, dtype=torch.long),)
torch_out = model(*torch_in)
torch.onnx.export(model, torch_in, onnx_path, verbose=True,
        export_params=True)

prog = onnx.load(onnx_path)
weights = prog.graph.initializer
for w in weights:
    print(w.name, numpy_helper.to_array(w).shape)