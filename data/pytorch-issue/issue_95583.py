import torch

t = torch.tensor([0., torch.nan, 2., torch.nan, 2., 1., 0., 1., 2., 0.])
print(t.unique(sorted=True, dim=None))
print(t.unique(sorted=True, dim=0))

tensor([1., nan, 2., nan, 0.])
tensor([0., nan, 2., nan, 0., 1., 2.])

tensor([0., 1., 2., nan, nan])
tensor([0., 1., 2., nan, nan])

tensor([0., 1., 2., nan, nan])
tensor([0., 1., 2., nan, nan])

tensor([0., 1., 2., nan])
tensor([0., 1., 2., nan])

tensor([0., nan, 2., nan, 0., 1., 2.])

import torch

t = torch.tensor([0., torch.nan, 2., torch.nan, 2., 1., 0., 1., 2., 0.])
print(t.unique(sorted=True, dim=None))

tensor([1., nan, 2., nan, 0.])

tensor([0., 1., 2., nan, nan])