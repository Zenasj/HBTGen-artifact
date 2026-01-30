def forward(self, inp1, inp2, inp3, inp4):
        temp1 = torch.fmin(input=inp4, other=inp3, out=inp2) #***
        temp2 = torch._C._special.special_entr(input=temp1, out=inp1) #***
        res1 = torch.prod(input=inp1) #***
        res2 = torch.full_like(fill_value=6, input=temp2, requires_grad=True)
        return res1, res2

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp1, inp2, inp3, inp4):
        res1 = torch.prod(input=inp1)
        temp1 = torch.fmin(input=inp4, other=inp3, out=inp2)
        temp2 = torch._C._special.special_entr(input=temp1, out=inp1)
        res2 = torch.full_like(fill_value=6, input=temp2, requires_grad=True)
        return res1, res2

model = Model().to(torch.device("cpu"))
inputs = {
    "inp1" : torch.rand([1, 1, 1, 1, 1, 1], dtype=torch.float32),
    "inp2" : torch.rand([2, 4], dtype=torch.float32),
    "inp3" : torch.rand([], dtype=torch.float32),
    "inp4" : torch.randint(-2147483648, 2147483647, [2], dtype=torch.int32)
}

ret_eager = model(**inputs)
ret_exported = torch.compile(model)(**inputs)

print('==== Check ====')
for r1, r2 in zip(ret_eager, ret_exported):
    if not torch.allclose(r1, r2, rtol=1e-2, atol=1e-3, equal_nan=True):
        print("r1: ",r1,"r2: ",r2)
        raise ValueError("Tensors are different.")

ret_eager = model(**{key: v.clone() for key, v in inputs.items()})