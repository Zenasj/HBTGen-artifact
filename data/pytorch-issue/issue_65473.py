import torch

try:
    lib_path = _get_extension_path("image")
    torch.ops.load_library(lib_path)
except (ImportError, OSError):
    pass