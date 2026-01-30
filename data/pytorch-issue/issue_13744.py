import torch.nn as nn

import torch

class MyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x):
        return self.sm(x)

model = MyModule()
model.train(False)
x = torch.randn(2, 3, 4)

torch.save(model, "/tmp/model.pt")

loaded_model = torch.load("/tmp/model.pt")

torch.onnx.export_to_pretty_string(loaded_model, x, None)