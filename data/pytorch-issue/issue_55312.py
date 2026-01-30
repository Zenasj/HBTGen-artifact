import torch
import torch.nn as nn

3
@torch.jit.script
def print_kernel_size(layer: nn.Conv1d):
    print(layer.kernel_size)