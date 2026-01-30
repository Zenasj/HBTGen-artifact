import torch

sparse_mult
tensor([[0.7071, 0.7071, 0.0000, 0.0000],
        [0.0000, 1.0000, 0.0000, 0.0000],
        [0.0000, 1.4142, 0.0000, 0.0000],
        [0.0000, 0.5774, 0.0000, 1.1547]])

norm_adj = torch.tensor([[1., 0., 0., 1.],
                         [0., 1., 0., 0.],
                         [0., 1., 1., 0.],
                         [0., 1., 0., 2.]])

idx = norm_adj.nonzero()
vals = norm_adj[norm_adj>0]
norm_adj = torch.sparse.FloatTensor(idx.t(),vals,torch.Size([4,4]))
norm_adj = norm_adj.coalesce()
print(norm_adj.to_dense())    
# print(norm_adj)
D = norm_adj.mm(torch.ones(4,1)).view(-1)
D = D ** -0.5
D = torch.diag(D)
print(D)
print('sparse_mult')
print(norm_adj.t().mm(D).t())
print('another dense')
print(norm_adj.to_dense().t().mm(D).t())
print('dense mult')
print(D.mm(norm_adj.to_dense()))