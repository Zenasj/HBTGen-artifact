import torch.nn as nn

m = nn.LazyBatchNorm1d(10)
ip_values = {'input': torch.randn([1,2,3])}
m(**ip_values) # This line causes the error

m(torch.randn([1,2,3]))
m(**ip_values) # Now this works