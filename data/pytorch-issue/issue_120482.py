import torch.nn as nn

import torch
from torch import nn


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.outer(**inputs)

compile_fn = torch.compile(CustomModel(), fullgraph=True)

shapes = [(2,1), (6,1), (4,1)]
for shape in shapes:
    vec1, vec2 = shape
    input_tensor1 = torch.randn(vec1)
    input_tensor2 = torch.randn(vec2)
    out_tensor = torch.empty(shape)
    res = compile_fn(
        {"input": input_tensor1, "vec2": input_tensor2, "out": out_tensor}
    )
    print(f"res: {res.shape}")
print("Test passed!")