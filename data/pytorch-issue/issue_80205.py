import torch.nn as nn

import torch

def get_fn():
    lower = -2.9
    upper = -2.7
    training = True
    def fn(input):
        fn_res = torch.nn.functional.rrelu(input, lower=lower, upper=upper, training=training)
        return fn_res
    return fn

input_tensor = torch.tensor([0.1250, 0.4313], dtype=torch.float64)
fn = get_fn()
try:
    input = input_tensor.clone().detach().to('cuda').requires_grad_()
    torch.autograd.gradcheck(fn, (input), check_sparse_nnz=False, atol=0.01, rtol=0.01, check_forward_ad=False, check_backward_ad=True, check_batched_grad=False)
except Exception as e:
    print(e)

# Jacobian mismatch for output 0 with respect to input 0,
# numerical:tensor([[1.0000, 0.0000],
#         [0.0000, 1.0000]], device='cuda:0', dtype=torch.float64)
# analytical:tensor([[0., 0.],
#         [0., 0.]], device='cuda:0', dtype=torch.float64)