import torch

from torch.testing._internal.common_utils import IS_FBCODE

if IS_FBCODE:
   cpp_compile_command = "..."