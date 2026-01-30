import torch
import torch.nn as nn
from copy import deepcopy
print("torch version: ",torch.__version__)
p0 =  torch.randn((), requires_grad=False)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.parameter.Parameter objects (with comments for shapes)
        self.p0 = p0


    def forward(self, v0_0):
        # v0_0: [], torch.float32
        v6_0 = torch.Tensor.sigmoid_(self.p0)
        v5_0 = torch.Tensor.atan2_(v0_0, other=self.p0)
        return v6_0, v5_0
    
inputs = {"v0_0": torch.randn(()).to(torch.device("cpu"))}
model = Model().to(torch.device("cpu"))
copied = deepcopy(inputs)
for k, v in inputs.items():
    inputs[k] = v.to(torch.device("cpu"))
print('==== Eager mode ====')
ret_eager = model(**inputs)

print('==== TorchComp mode ====')
ret_exported = torch.compile(model)(**copied)
print(ret_eager, ret_exported)
print('==== Check ====')
for r1, r2 in zip(ret_eager, ret_exported):
    if not torch.allclose(r1, r2, rtol=1e-2, atol=1e-3, equal_nan=True):
        print("r1: ",r1,"r2: ",r2)
        raise ValueError("Tensors are different.")
print('OK!')