import torch

from torch import einsum, tensor, matmul
t = tensor([[[0., 1.],
             [2., 3.]]], device='mps')

# result from CPU is correct:
einsum('b i d, b j d -> b i j', t.cpu(), t.cpu())
# tensor([[[ 1.,  3.],
#          [ 3., 13.]]])

# first result from MPS is wrong:
einsum('b i d, b j d -> b i j', t, t)
# tensor([[[ 2.,  3.],
#          [ 6., 11.]]], device='mps:0')

# subsequent results from MPS are correct:
einsum('b i d, b j d -> b i j', t, t)
# tensor([[[ 1.,  3.],
#          [ 3., 13.]]], device='mps:0')

einsum('b i d, b j d -> b i j', t, t)
# tensor([[[ 1.,  3.],
#          [ 3., 13.]]], device='mps:0')

# btw this einsum is equivalent to the following matmul:
matmul(t, t.transpose(1, 2))
# tensor([[[ 1.,  3.],
#          [ 3., 13.]]], device='mps:0')
# in other words a matmul over these:
# tensor([[[0., 1.],
#          [2., 3.]]]) *
# tensor([[[0., 2.],
#          [1., 3.]]]) =
# tensor([[[0*0+1*1, 2*0+3*1],
#          [2*0+3*1, 2*2+3*3]]])

# q.shape
# torch.Size([16, 4096, 40])
# k.shape
# torch.Size([16, 4096, 40])
einsum('b i d, b j d -> b i j', q, k)

# attn.shape
# torch.Size([16, 4096, 4096])
# v.shape
# torch.Size([16, 4096, 40])
einsum('b i j, b j d -> b i d', attn, v)