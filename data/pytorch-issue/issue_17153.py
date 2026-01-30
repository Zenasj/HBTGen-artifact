import torch.nn as nn

reproduce.py
import torch


class MyModel(torch.nn.Module):
    def forward(self, x):
        # return torch.randn(3, 3)  # OK
        return torch.randn(*x.size())   # NG


model = MyModel()

dummy_input = torch.randn(3, 3)
torch.onnx.export(model, dummy_input, "mymodel.onnx")

expected.py
# Nothing to show. mymodel.onnx is successfully generated.