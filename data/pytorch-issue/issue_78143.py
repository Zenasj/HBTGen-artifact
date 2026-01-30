import torch

input_tensor = torch.rand([10, 3, 5], dtype=torch.complex64)
batch1_tensor = torch.randint(-4, 2, [10, 3, 4], dtype=torch.int8)
batch2_tensor = torch.randint(-4, 1, [10, 4, 5], dtype=torch.int8)

input = input_tensor.clone()
batch1 = batch1_tensor.clone()
batch2 = batch2_tensor.clone()
res1 = torch.baddbmm(input, batch1, batch2, )
# Normal Pass

input = input_tensor.clone().requires_grad_()
batch1 = batch1_tensor.clone()
batch2 = batch2_tensor.clone()
res2 = torch.baddbmm(input, batch1, batch2, )
# RuntimeError: isDifferentiableType(variable.scalar_type())INTERNAL ASSERT FAILED at "/Users/distiller/project/pytorch/torch/csrc/autograd/functions/utils.h":65, please report a bug to PyTorch.