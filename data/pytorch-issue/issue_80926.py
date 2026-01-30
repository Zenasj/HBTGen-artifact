import torch.nn as nn

py
import torch

def get_fn():
    arg_class = torch.nn.Softsign()
    def fn(input):
        fn_res = arg_class(input)
        return fn_res
    return fn
fn = get_fn()
input_tensor = torch.tensor([-0.3606-0.7886j, -0.4246-0.1352j], dtype=torch.complex128)

try:
    input = input_tensor.clone().detach().to('cpu').requires_grad_()
    torch.autograd.gradgradcheck(fn, (input), atol=0.01, rtol=0.01, check_fwd_over_rev=True, check_rev_over_rev=False, check_undefined_grad=False)
except Exception as e:
    print('fwdrev fail')
    print(e)

try:
    input = input_tensor.clone().detach().to('cpu').requires_grad_()
    torch.autograd.gradgradcheck(fn, (input), atol=0.01, rtol=0.01, check_fwd_over_rev=False, check_rev_over_rev=True, check_undefined_grad=False)
    print('revrev success')
except Exception as e:
    print(e)