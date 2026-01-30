py
import torch

def fn(input):
    fn_res = torch.sgn(input, )
    return fn_res

input_tensor = torch.tensor(-3.8766-1.1128j, dtype=torch.complex128)

input = input_tensor.clone().requires_grad_()
torch.autograd.gradcheck(fn, (input), atol=0.01, rtol=0.01, check_forward_ad=False, check_backward_ad=True)
print('rev success')

try:
    input = input_tensor.clone().requires_grad_()
    torch.autograd.gradcheck(fn, (input), atol=0.01, rtol=0.01, check_forward_ad=True, check_backward_ad=False)
except Exception as e:
    print(e)