import torch.nn as nn

import os
import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y = x.T @ x
        v, w = torch.linalg.eigh(y)
        y = self.fc1(w)
        y = self.relu(y)
        y = self.fc2(y)
        y = 0.5 * (self.sigmoid(y) + 1.0)
        return y


with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model().to(device=device)
    example_inputs = (torch.randn((8, 10), device=device, requires_grad=False),)
    batch_dim = torch.export.Dim("batch", min=1, max=1024)
    so_path = torch._export.aot_compile(
        model,
        example_inputs,
        # Specify the first dimension of the input x as dynamic
        dynamic_shapes={"x": {0: batch_dim}},
        # Specify the generated shared library path
        options={"aot_inductor.output_path": os.path.join(os.getcwd(), "model2.so")},
    )