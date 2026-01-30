import torch
i = torch.LongTensor([
            [0, 1, 2],
            [2, 1, 0],
         ])
v = torch.FloatTensor([1, 1, 1])
D = torch.sparse.FloatTensor(i, v, torch.Size([3,3]))
Dc = D.coalesce()

ii = Dc._indices()
ii[0,0] = 2
ii

Dc._indices()

Dc.is_coalesced()

Dc[0,0] = 2