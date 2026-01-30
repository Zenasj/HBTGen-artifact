import torch
def fn(input):
    fn_res = torch.angle(input)
    return fn_res

input = torch.rand([3], dtype=torch.complex128, requires_grad=True)
torch.autograd.gradcheck(fn, (input), check_forward_ad=False, check_backward_ad=True)
print('backward mode gradcheck PASS!')
torch.autograd.gradcheck(fn, (input), check_forward_ad=True, check_backward_ad=False)

# backward mode gradcheck PASS!
# 
# GradcheckError: While considering the imaginary part of complex inputs only, Jacobian computed with forward mode mismatch for output 0 with respect to input 0,
# numerical:tensor([[0.7133, 0.0000, 0.0000],
#         [0.0000, 0.7700, 0.0000],
#         [0.0000, 0.0000, 0.7393]], dtype=torch.float64)
# analytical:tensor([[-0.7133,  0.0000,  0.0000],
#         [ 0.0000, -0.7700,  0.0000],
#         [ 0.0000,  0.0000, -0.7393]], dtype=torch.float64,
#        grad_fn=<CopySlices>)