import torch
a=torch.randint(-5, 10, [5, 5], device='cuda')
b=torch.randint(-5, 10, [5, 5], device='cuda')
a_s=a.to_sparse()
b_s=b.to_sparse()
a.mul_(b)
a_s.mul_(b_s)
print( "success" if torch.allclose(a, a_s.to_dense()) else "fail")

import torch

a=torch.randint(-5, 10, [5, 5], device='cuda')
b=torch.randint(-5, 10, [5, 5], device='cuda')

a_s=a.to_sparse()
i = a_s._indices()
i=i.transpose(0, 1)
i=i.contiguous()
i=i.transpose(0, 1)
print(i.is_contiguous())
a_s=torch.sparse_coo_tensor(indices=i, values=a_s._values())

b_s=b.to_sparse()
i = b_s._indices()
i=i.transpose(0,1)
i=i.contiguous()
i=i.transpose(0,1)

print(i.is_contiguous())
b_s=torch.sparse_coo_tensor(indices=i, values=b_s._values())

a.mul_(b)
a_s.mul_(b_s)
print( "success" if torch.allclose(a, a_s.to_dense()) else "fail")