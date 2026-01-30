import torch
sp_val = torch.rand(10).to('cuda:1')
print(sp_val, sp_val.device)

sp_ind = torch.tensor([[0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9]], device='cuda:1')
print(sp_ind)

x = torch.sparse_coo_tensor(sp_ind, sp_val, (10, 10))

import torch
i=torch.tensor(([0], [2]), device="cuda:0", dtype=torch.long)
v=torch.tensor([1.], device="cuda:1")           
t=torch.sparse_coo_tensor(i, v, dtype=torch.float64)    
print(t.device) # CPU