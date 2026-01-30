import torch

torch.typename(xx)
'torch.FloatTensor'

torch.__version__
Out[16]: '1.7.1'

torch.cat(( torch.tensor([]), torch.tensor([1,2,3], dtype=torch.long)))
Out[22]: tensor([1., 2., 3.])