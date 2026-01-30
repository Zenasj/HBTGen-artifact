import torch

out_tensor = torch.empty(3, 3, dtype=torch.float, requires_grad=True)
out_tensor.fill_(-1)
print(out_tensor)

import torch

out_tensor = torch.empty(3, 3, dtype=torch.float, requires_grad=True)
out_tensor = out_tensor.cuda()
out_tensor.fill_(-1)
print(out_tensor)

out_tensor = torch.empty(3, 3, dtype=torch.float, requires_grad=True, device='cuda')
out_tensor.fill_(-1)