import torch

get_top = torch.argsort(similarity, dim=1)
get_top = torch.argsort(similarity, dim=1, descending=True)

In [33]: a = torch.tensor([3, float('NaN'), 2])

In [34]: torch.argsort(a, dim=0)
Out[34]: tensor([0, 1, 2]) # wrong!

In [36]: a = torch.tensor([1, float('NaN'), 2])

In [37]: torch.argsort(a, dim=0, descending=True)
Out[37]: tensor([0, 1, 2]) # wrong!

In [38]: a = numpy.load("log")

In [39]: a = torch.tensor(a)

In [40]: b = a[1:]

In [41]: b
Out[41]: tensor([ 0.0791, -0.0522,  0.0226,  ..., -0.0267,  0.1027,  0.0064])

In [49]: print(torch.argsort(b, dim=0, descending=True).narrow(0,0,8))
tensor([ 49, 117,  46,  12, 143,  16,  19, 327])

In [50]: print(torch.argsort(b, dim=0).narrow(0,b.shape[0]-8,8))
tensor([327,  19,  16, 143,  12,  46, 117,  49])