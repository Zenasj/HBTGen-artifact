import math, io
import torch


def getExportImportCopy(m):
    buffer = io.BytesIO()
    torch.jit.save(m, buffer)
    buffer.seek(0)
    return torch.jit.load(buffer)

def allSum(vs):
    return sum(math.log(i + 2) * v.sum() for i, v in enumerate(vs) if v is not None)

def func(a, b):
    return a * b / (a - 2 * b) + b

torch.set_printoptions(precision=30)

tensor = torch.tensor
recording_inputs = [tensor([0.55619788169860839844], dtype=torch.float32, requires_grad=True),
                    tensor([0.25947844982147216797], dtype=torch.float32, requires_grad=True)]

ge1 = torch.jit.trace(func, recording_inputs, optimize=True)
ge2 = getExportImportCopy(ge1)

outputs_ge1 = ge1(*recording_inputs)
outputs_ge2 = ge2(*recording_inputs)
print(outputs_ge1, outputs_ge1.dtype)
print(outputs_ge2, outputs_ge2.dtype)

grad_ge1 = torch.autograd.grad(allSum(outputs_ge1), recording_inputs)
grad_ge2 = torch.autograd.grad(allSum(outputs_ge2), recording_inputs)
print(grad_ge1)
print(grad_ge2)