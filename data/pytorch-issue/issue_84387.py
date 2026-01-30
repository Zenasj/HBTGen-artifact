import torch

# In some circumstances, we will be tracing in a situation where a tensor
# is *statically* known to be a constant (currently, this only happens if
# you run torch.tensor; deterministic factory functions like torch.arange
# don't get this treatment).  When the tensor in question is small, it's
# helpful to due constant propagation in case we call item() (in which
# case we can return the constant value that is known, rather than give
# an error.)