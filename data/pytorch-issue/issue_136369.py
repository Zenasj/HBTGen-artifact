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
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

with torch.no_grad():
    device="cuda:1"
    model = Model().to(device=device)
    inp=torch.randn(8, 10, device=device)
    batch_dim = torch.export.Dim("batch", min=1, max=1024)

    so_path = torch._export.aot_compile(
        model,
        (inp,),
        dynamic_shapes={"x": {0: batch_dim}},
        options={"aot_inductor.output_path": os.path.join(os.getcwd(), "model.so")},
    )

    torch._export.aot_load(
        so_path,
        device=device
    )