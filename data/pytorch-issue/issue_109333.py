import torch

In [1]: a = torch.ones(2, 3, dtype=torch.float)

In [2]: b = torch.ones(1, 1, dtype=torch.float)

In [3]: torch.cosine_similarity(a, b)
Out[3]: tensor([1.7321, 1.7321])