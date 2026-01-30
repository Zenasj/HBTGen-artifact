import torch

from torch.utils.cpp_extension import ROCM_HOME
print("ROCM_HOME:'{}', type(ROCM_HOME): '{}'".format(ROCM_HOME, type(ROCM_HOME)))