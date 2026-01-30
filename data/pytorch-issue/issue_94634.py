py
import torch
from torch.autograd.functional import jacobian
from torch.func import jacrev, jacfwd

torch.manual_seed(420)

input_tensor = torch.ones(1, 3)
mask = torch.ones(1, 3)
tensor = torch.ones(3, 4)

def func(input_tensor, mask, tensor):
    output_tensor = torch.masked_scatter(input_tensor, mask, tensor)
    return output_tensor

func(input_tensor, mask, tensor)
# succeed

func(input_tensor.clone().requires_grad_(), mask.clone().requires_grad_(), tensor.clone().requires_grad_())
# succeed

jacobian(func, (input_tensor, mask, tensor), vectorize=True, strategy="reverse-mode")
# RuntimeError: masked_select: expected BoolTensor or ByteTensor for mask

jacrev(func)(input_tensor, mask, tensor)
# RuntimeError: masked_select: expected BoolTensor or ByteTensor for mask