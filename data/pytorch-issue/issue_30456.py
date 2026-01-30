import torch

from torch.utils.cpp_extension import load
foo = load(name='foo', sources='foo.cpp')