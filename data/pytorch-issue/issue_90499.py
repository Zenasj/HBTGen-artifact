import torch
import functorch

m = torch.tensor([[0.30901699437494745, 0.9510565162951535], [-0.9510565162951535, 0.30901699437494745]], dtype=torch.float64)

print((functorch.jacfwd(torch.linalg.eigvals)(m)))
print()

print((functorch.jacrev(torch.linalg.eigvals)(m)))
print()

# tensor([[[0.5000+0.0000j, 0.0000+0.5000j],
#          [0.0000-0.5000j, 0.5000+0.0000j]],

#         [[0.5000+0.0000j, 0.0000-0.5000j],
#          [0.0000+0.5000j, 0.5000+0.0000j]]], dtype=torch.complex128)

# tensor([[[0.5000, 0.0000],
#          [0.0000, 0.5000]],

#         [[0.5000, 0.0000],
#          [0.0000, 0.5000]]], dtype=torch.float64)

import torch
import functorch

m = torch.tensor([[0.30901699437494745, 0.9510565162951535], [-0.9510565162951535, 0.30901699437494745]], dtype=torch.float64)

# OK
functorch.jacfwd(torch.linalg.eig)(m)

# RuntimeError: vmap over torch.allclose isn't supported yet. Please voice your support over at github.com/pytorch/functorch/issues/275
functorch.jacrev(torch.linalg.eig)(m)