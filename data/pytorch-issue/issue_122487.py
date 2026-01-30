import torch.nn as nn

import os
import hashlib
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        return x

class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(x)
        x = self.sigmoid(x)
        return x

with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    example_inputs=(torch.randn(8, 10, device=device),)
    batch_dim = torch.export.Dim("batch", min=1, max=1024)

    # Model 1 ##
    model = Model().to(device=device)
    so_path = torch._export.aot_compile(
        model,
        example_inputs,
        # Specify the first dimension of the input x as dynamic
        dynamic_shapes={"x": {0: batch_dim}},
        # Specify the generated shared library path
        options={"aot_inductor.output_path": "/tmp/exported/model.so"},
    )
    res1 = model(*example_inputs)
    hashval1 = hashlib.md5(open(so_path, 'rb').read()).hexdigest()

    ## Model 2 ## 
    model2 = Model2().to(device=device)
    so_path = torch._export.aot_compile(
        model2,
        example_inputs,
        # Specify the first dimension of the input x as dynamic
        dynamic_shapes={"x": {0: batch_dim}},
        # Specify the generated shared library path
        options={"aot_inductor.output_path": "/tmp/exported/model.so"},
    )
    res2 = model2(*example_inputs)
    hashval2 = hashlib.md5(open(so_path, 'rb').read()).hexdigest()

    assert res1.sum() != res2.sum(), "results are different"
    assert hashval1 != hashval2, "hash musht be different"