import torch

def fn(input):
    offset = 0
    fn_res = torch.diagonal(input, offset=offset, )
    return fn_res

input= torch.rand([0, 1], dtype=torch.complex128, requires_grad=True)
torch.autograd.gradcheck(fn, (input), check_forward_ad=True, check_backward_ad=False)
# RuntimeError: !this_view_meta->has_fw_view()INTERNAL ASSERT FAILED at "/Users/distiller/project/pytorch/torch/csrc/autograd/autograd_meta.cpp":186, please report a bug to PyTorch. Expected the output of forward differentiable view operations to have the tangent have the same layout as primal