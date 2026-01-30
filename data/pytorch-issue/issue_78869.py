import torch

def fn(input):
    res = torch.mean(input, dtype=torch.int8)
    return res

input_tensor = torch.rand([5, 5, 5], dtype=torch.complex128)

input = input_tensor.clone().detach().to('cuda')
res = fn(input)

input = input_tensor.clone().detach().to('cuda').requires_grad_()
res_grad = fn(input)
# RuntimeError: isDifferentiableType(variable.scalar_type())INTERNAL ASSERT FAILED at "../torch/csrc/autograd/functions/utils.h":65, please report a bug to PyTorch