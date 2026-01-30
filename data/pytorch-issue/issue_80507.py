import torch

input = torch.tensor([1.0+1.0j], dtype=torch.complex128, requires_grad=True)
torch.autograd.gradcheck(torch.diff, (input), check_forward_ad=True, check_backward_ad=False)

# !this_view_meta->has_fw_view()INTERNAL ASSERT FAILED at "/opt/conda/conda-bld/pytorch_1646756402876/work/torch/csrc/autograd/autograd_meta.cpp":186, please report a bug to PyTorch. Expected the output of forward differentiable view operations to have the tangent have the same layout as primal