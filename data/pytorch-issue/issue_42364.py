import torch

torch.max(torch.randn(2,3).cuda(1), dim=0, out=(torch.randn(6).cuda(1)[::2], torch.randn(3).long().cuda(1)))
torch.return_types.max_out(
values=tensor([0.1383, 1.9499, 1.2522], device='cuda:1'),
indices=tensor([0, 1, 1], device='cuda:1'))