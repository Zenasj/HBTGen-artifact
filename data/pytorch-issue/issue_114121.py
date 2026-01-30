import torch
ld = torch.randn(1,1,1,dtype=torch.float64)
pivots = torch.randint(9223372036854775807,(1,1),dtype=torch.int64)
b = torch.randn(1,1,1,1,dtype=torch.float64)
result = torch.linalg.ldl_solve(ld,pivots,b,hermitian=False)