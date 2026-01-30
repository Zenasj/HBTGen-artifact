import torch
[torch.Tensor(), None, None].count(None)

#Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#  File ".../lib/python2.7/site-packages/torch/tensor.py", line 312, in __eq__
#    return self.eq(other)
#TypeError: eq received an invalid combination of arguments - got (NoneType), but expected one of:
# * (float value)
#      didn't match because some of the arguments have invalid types: (NoneType)
# * (torch.FloatTensor other)
#      didn't match because some of the arguments have invalid types: (NoneType)

[torch.Tensor([1, 2]), 1, 2].count(1)
#File "/home/mscho/vadim/.wigwam/prefix/python/lib/python2.7/site-packages/torch/tensor.py", line # 168, in __bool__    " containing more than one value is ambiguous")

# what should be the semantics here? 
[torch.Tensor([1]), 1, 2].count(1)
#2