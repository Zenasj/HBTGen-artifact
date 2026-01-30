py
import torch
def fn(_input_tensor):
    fn_res = _input_tensor.abs()
    return fn_res

_input_tensor_tensor = torch.tensor([[[-3.4118+4.8582j,  1.4920-0.7004j]]], dtype=torch.complex128, requires_grad=True)

_input_tensor = _input_tensor_tensor.clone().requires_grad_()
torch.autograd.gradgradcheck(fn, (_input_tensor), atol=0.01, rtol=0.01, check_fwd_over_rev=False, check_rev_over_rev=True, check_undefined_grad=False)
print('revrev succeed')

try:
    _input_tensor = _input_tensor_tensor.clone().requires_grad_()
    torch.autograd.gradgradcheck(fn, (_input_tensor), atol=0.01, rtol=0.01, check_fwd_over_rev=True, check_rev_over_rev=False, check_undefined_grad=False)
except Exception as e:
    print(e)
    print('fwdrev error')