import torch

from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor

data_ptr = 0 if isinstance(tensor, (FakeTensor, FunctionalTensor)) else tensor.data_ptr()