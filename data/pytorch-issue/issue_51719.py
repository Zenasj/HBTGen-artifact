import torch
a = torch.rand(60)
b = a.unflatten(0, (10, -1))

#   File "/specific/netapp5_2/gamir/lab/vadim/prefix/miniconda/lib/python3.8/site-packages/torch/tensor.py", line 861, in unflatten
#    return super(Tensor, self).unflatten(dim, sizes, names)
#RuntimeError: unflatten: Provided sizes [10, -1] don't multiply up to the size of dim 0 (60) in the input tensor