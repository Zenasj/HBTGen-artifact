py
import torch
from torch.autograd.functional import jacobian
from torch.autograd import gradcheck

input_data = torch.tensor([0, 1, 2, 3], dtype=torch.float)

def func(input_data):
    output_data = torch.clamp(input_data, min=1, max=1)
    return output_data

print(jacobian(func, input_data, vectorize=True, strategy="forward-mode"))
# tensor([[0., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])

gradcheck(func, input_data.requires_grad_())
# torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 0,
# numerical:tensor([[0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])
# analytical:tensor([[0., 0., 0., 0.],
#         [0., 1., 0., 0.],
#         [0., 0., 0., 0.],
#         [0., 0., 0., 0.]])