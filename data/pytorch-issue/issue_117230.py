import torch

torch.kaiser_window(2, device=torch.device('cpu'), dtype=torch.bfloat16, layout=torch.strided, requires_grad=False)