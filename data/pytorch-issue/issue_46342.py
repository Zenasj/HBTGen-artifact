import torch
from torch import vmap
from torch.autograd import grad

N = 2
x = torch.randn(N, 3, requires_grad=True)
y = torch.randn(N, 3)
distances = torch.cdist(x, y)
flat_distances = distances.view(-1)

basis_vectors = torch.eye(N * N)
jacobian, = vmap(grad, (None, None, 0))(flat_distances, x, basis_vectors)
print(jacobian)