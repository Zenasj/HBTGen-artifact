import torch.nn as nn

import torch
from torch_geometric.nn import SAGEConv

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(8, 16)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

model = MyModel()
x = torch.randn(3, 8)
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

torch.onnx.export(
    model,
    (x, edge_index),
    "model.onnx",
    opset_version=16,
    input_names=["x", "edge_index"],
    dynamic_axes={
        "x": {0: "n_nodes"},
        "edge_index": {1: "n_edges"}
    }
)

torch.onnx.export(
    model,
    ((x, x), edge_index),
    "model.onnx",
    opset_version=16,
    input_names=["x", "edge_index"],
    dynamic_axes={
        "x": {0: "n_nodes"},
        "edge_index": {1: "n_edges"}
    }
)

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(8, 16)
        self.conv2 = SAGEConv(16, 8)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x