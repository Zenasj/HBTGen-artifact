import torch
import torch.nn as nn

{python}
import numpy as np
import sys
print("python={}".format(sys.version_info) )
print("torch.__version__={}".format(torch.__version__))
# python=sys.version_info(major=3, minor=7, micro=11, releaselevel='final', serial=0)
# torch.__version__=1.10.0

# Input shapes (N=3, C=3)
_ones = torch.ones(N,1)
_zeros = torch.zeros(N,1)
X = torch.cat([_ones, _zeros, _zeros], dim=1) # respond all sample as class 0
# torch.Size([3, 3])

# Target shapes (N=3,) with ignore_index=-100
Y = torch.tensor([0,1,-100]).type(torch.LongTensor)
print(Y) # tensor([   0,    1, -100])

weight=torch.ones(3) # for three classes
_ = torch.nn.functional.cross_entropy(X,Y, weight=weight, ignore_index=-100, label_smoothing=0,   reduction='mean') # label_smoothing=0
_ = torch.nn.functional.cross_entropy(X,Y, weight=None,   ignore_index=-100, label_smoothing=0.1, reduction='mean') # wieght=None
_ = torch.nn.functional.cross_entropy(X,Y, weight=weight, ignore_index=-100, label_smoothing=0.1, reduction='sum') # reduction='sum'
_ = torch.nn.functional.cross_entropy(X,Y, weight=weight, ignore_index=-100, label_smoothing=0.1, reduction='none') # reduction='none'
_ = torch.nn.functional.cross_entropy(X,Y, weight=weight, ignore_index=-100, label_smoothing=0.1, reduction='mean') # error is only here