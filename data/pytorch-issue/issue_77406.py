import torch

a = torch.zeros(4)
b = torch.tensor([[0.],[1.],[2.],[3.]])
i = torch.tensor([3,2,1,0])

a.index_put([i],b)
# RuntimeError: shape mismatch: value tensor of shape [4, 1] cannot be broadcast to indexing result of shape [4]

a.index_put([i],b,accumulate=True)
# RuntimeError: shape mismatch: value tensor of shape [4, 1] cannot be broadcast to indexing result of shape [4]

a.to('cuda').index_put([i.to('cuda')],b.to('cuda'))
# RuntimeError: shape mismatch: value tensor of shape [4, 1] cannot be broadcast to indexing result of shape [4]

a.to('cuda').index_put([i.to('cuda')],b.to('cuda'),accumulate=True)
# tensor([3., 2., 1., 0.], device='cuda:0')

a.index_put([i],b.squeeze())
# tensor([3., 2., 1., 0.])

a.index_put([i],b.squeeze(),accumulate=True)
# tensor([3., 2., 1., 0.])

a.to('cuda').index_put([i.to('cuda')],b.squeeze().to('cuda'))
# tensor([3., 2., 1., 0.], device='cuda:0')

a.to('cuda').index_put([i.to('cuda')],b.squeeze().to('cuda'),accumulate=True)
# tensor([3., 2., 1., 0.], device='cuda:0')