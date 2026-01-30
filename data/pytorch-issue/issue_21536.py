import torch
import torch.nn as nn
 
import kornia
 
@torch.jit.script
def op_script(input: torch.Tensor, height: int,
              width: int) -> torch.Tensor:
    return kornia.normalize_pixel_coordinates(input, height, width)
 
class MyTestModule(nn.Module):
    def __init__(self):
        super(MyTestModule, self).__init__()
 
    def forward(self, input):
        height, width = input.shape[-2:]
        return op_script(input, height, width)
 
 
my_test_op = MyTestModule()
op_traced = torch.jit.trace(my_test_op, torch.rand(1,2,4,4))

import torch
import torch.nn as nn
from torch.testing import assert_allclose

import kornia

@torch.jit.script
def op_script(input: torch.Tensor, height: int,
              width: int) -> torch.Tensor:
    return kornia.normalize_pixel_coordinates(input, height, width)

class MyTestModule(nn.Module):
    def __init__(self):
        super(MyTestModule, self).__init__()

    def forward(self, input): 
        height, width = input.shape[-2:]
        return op_script(input, height, width)


my_test_op = MyTestModule()
op_traced = torch.jit.trace(my_test_op, torch.rand(1,4,4,2))

# create points grid

height, width = 5, 5
points = kornia.create_meshgrid(
    height, width, normalized_coordinates=False) # 1xHxWx2

# we expect that the traced function generalises with different
# input shapes. Ideally we might want to infer to traced the h and w.
assert_allclose(op_traced(points),
    kornia.normalize_pixel_coordinates(points, height, width))

@torch.jit.script
def op_script(input, height, width):
    return kornia.normalize_pixel_coordinates(input, int(height), int(width))

@torch.jit.script
def op_script(input, height, width):
    return kornia.normalize_pixel_coordinates(input, int(height), int(width))

class MyTestModule(nn.Module):
    def __init__(self):
        super(MyTestModule, self).__init__()

    def forward(self, input):
        height, width = input.shape[-2:]
        if not torch._C._get_tracing_state():
            height = torch.tensor(height)
            width = torch.tensor(width)

        return op_script(input, height, width)


my_test_op = MyTestModule()
op_traced = torch.jit.trace(my_test_op, torch.rand(1,4,4,2))