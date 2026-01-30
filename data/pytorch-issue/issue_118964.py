import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.p0 = torch.nn.parameter.Parameter(param, requires_grad=False) # [7, 3]

    def forward(self, i1, i2):
        r1, r2 = torch._C._linalg.linalg_qr(A=i1)
        without_this_no_error = torch.pow(exponent=i2, input=self.p0, out=i1)
        return  r1, r2

param = torch.rand([7,3], dtype=torch.float32)
model = Model().to(torch.device("cpu"))
inputs = {
    "i1": torch.rand([7,3], dtype=torch.float32),
    "i2": torch.rand([7,3], dtype=torch.float32)
}

ret_eager = model(**inputs)
ret_exported = torch.compile(model)(**inputs)

for r1, r2 in zip(ret_eager, ret_exported):
    if not torch.allclose(r1, r2, rtol=1e-2, atol=1e-3, equal_nan=True):
        print("r1: ",r1,"r2: ",r2)
        raise ValueError("Tensors are different.")