import torch

# Run on system with and without Ninja

from torch.utils.cpp_extension import verify_ninja_availability
verify_ninja_availability()   # Does not return bool