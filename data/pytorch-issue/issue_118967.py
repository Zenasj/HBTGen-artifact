import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, i1, i2):
        r1 = torch.mean(input=i1)
        r2 = torch.asinh(input=i2, out=i1)
        return  r1, r2

model = Model().to(torch.device("cpu"))
inputs = {
    "i1": torch.rand([6], dtype=torch.float32),
    "i2": torch.randint(-128, 127, [6], dtype=torch.int8)
}

ret_eager = model(**inputs)
ret_exported = torch.compile(model)(**inputs)

for r1, r2 in zip(ret_eager, ret_exported):
    if not torch.allclose(r1, r2, rtol=1e-2, atol=1e-3, equal_nan=True):
        print("r1: ",r1,"r2: ",r2)
        raise ValueError("Tensors are different.")

raise ValueError("Tensors are different.")