import torch
a = torch.rand([2, 2])
a.triu(diagonal = 1)
# succeed
a.triu(k = 1)
# TypeError: triu() got an unexpected keyword argument 'k'