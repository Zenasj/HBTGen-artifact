import torch

py
batch_shape = (1, 2, 1, 3, 1)
sample_shape = (4,)
cardinality = 2
logits = torch.randn(batch_shape + (cardinality,))
dist.Categorical(logits=logits).sample(sample_shape)
# RuntimeError: invalid argument 2: view size is not compatible with
#   input tensor's size and stride (at least one dimension spans across
#   two contiguous subspaces). Call .contiguous() before .view().
#   at ../aten/src/TH/generic/THTensor.cpp:203