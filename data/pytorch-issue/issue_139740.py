import torch

torch.manual_seed(0)
op = torch.ops.aten._softmax_backward_data
grad_output = torch.ones(3, 3, 3)
temp = torch.randn(3, 10, 3)
out = temp[:, :3, :]
out = out.contiguous()
print(out.is_contiguous())
grad_input = op(grad_output, out, 1, torch.float32)
print(grad_input)