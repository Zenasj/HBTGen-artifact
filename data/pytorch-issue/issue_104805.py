import torch

# primals_1 is a (sym)int, primals_2 and primals_3 are tensors
def forward(self, primals_1, primals_2, primals_3):
    mul = torch.ops.aten.mul.Tensor(primals_2, 0.4);  primals_2 = None
    sub = torch.ops.aten.sub.Tensor(primals_3, mul);  primals_3 = mul = None
    return [sub, primals_1]