import torch

torch.tensor(5, device='cuda:0') + torch.tensor((1, 1), device='cuda:1')

torch.tensor(5, device='cuda:0') + torch.tensor((1, 1), device='cuda:1')

torch.tensor(2, device='cuda') + torch.tensor((3, 5))

TORCH_CHECK(false, 
"expected device ", op.device,
" but got device ", op.tensor.device());