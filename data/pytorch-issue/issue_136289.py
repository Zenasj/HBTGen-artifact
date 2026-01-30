import os
import sys
import time

import torch
import torch.nn as nn
from torch.distributed._tensor import Shard
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

import unittest
import torch
from torch._dynamo.utils import detect_fake_mode
from torch._functorch.aot_autograd import aot_export_joint_simple
from typing import Sequence, Any

class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10, 3200)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(3200, 1600)
        self.in_proj2 = nn.Linear(1600, 500)
        self.out_proj2 = nn.Linear(500, 100)

    def forward(self, x):
        x = self.out_proj(self.relu(self.in_proj(x)))
        x = self.relu(x)
        x = self.out_proj2(self.relu(self.in_proj2(x)))
        return x


# create a device mesh based on the given world_size.
_world_size = int(os.environ["WORLD_SIZE"])

device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))
_rank = device_mesh.get_rank()


print(f"Starting PyTorch TP example on rank {_rank}.")
assert (
    _world_size % 2 == 0
), f"TP examples require even number of GPUs, but got {_world_size} gpus"


# # create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
tp_model = ToyModel().to("cuda")


# Custom parallelization plan for the model
tp_model = parallelize_module(
    module=tp_model,
    device_mesh=device_mesh,
    parallelize_plan={
        "in_proj": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj": RowwiseParallel(output_layouts=Shard(0)),
        "in_proj2": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj2": RowwiseParallel(output_layouts=Shard(0)),
    },
)
torch.manual_seed(0)
inp = torch.rand(20, 10, device="cuda")
python_result = tp_model(inp)

def custom_backend(gm: torch.fx.GraphModule, sample_inputs: Sequence[Any], **kwargs: Any):
    fake_mode = detect_fake_mode(sample_inputs)
    with unittest.mock.patch.object(fake_mode, "allow_non_fake_inputs", True), fake_mode:
        torch_inputs = [input for input in sample_inputs if isinstance(input, torch.Tensor)]
    gm = aot_export_joint_simple(
        gm,
        torch_inputs,
        trace_joint=False,
        )
    return gm

tp_model = torch.compile(
    tp_model,
    backend=custom_backend,
    dynamic=False,
)
custom_backend_result = tp_model(inp)

import os
import sys
import time

import torch
import torch.nn as nn
from torch.distributed._tensor import Shard
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch._dynamo.backends.common import aot_autograd

import unittest
import torch
from torch._dynamo.utils import detect_fake_mode
from torch._functorch.aot_autograd import aot_export_joint_simple
from typing import Sequence, Any

class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(10, 3200)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(3200, 1600)
        self.in_proj2 = nn.Linear(1600, 500)
        self.out_proj2 = nn.Linear(500, 100)

    def forward(self, x):
        x = self.out_proj(self.relu(self.in_proj(x)))
        x = self.relu(x)
        x = self.out_proj2(self.relu(self.in_proj2(x)))
        return x


# create a device mesh based on the given world_size.
_world_size = int(os.environ["WORLD_SIZE"])

device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(_world_size,))
_rank = device_mesh.get_rank()


print(f"Starting PyTorch TP example on rank {_rank}.")
assert (
    _world_size % 2 == 0
), f"TP examples require even number of GPUs, but got {_world_size} gpus"


# # create model and move it to GPU - init"cuda"_mesh has already mapped GPU ids.
tp_model = ToyModel().to("cuda")


# Custom parallelization plan for the model
tp_model = parallelize_module(
    module=tp_model,
    device_mesh=device_mesh,
    parallelize_plan={
        "in_proj": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj": RowwiseParallel(output_layouts=Shard(0)),
        "in_proj2": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj2": RowwiseParallel(output_layouts=Shard(0)),
    },
)
torch.manual_seed(0)
inp = torch.rand(20, 10, device="cuda")
python_result = tp_model(inp)

def custom_backend(gm: torch.fx.GraphModule, sample_inputs: Sequence[Any], **kwargs: Any):
    fake_mode = detect_fake_mode(sample_inputs)
    with unittest.mock.patch.object(fake_mode, "allow_non_fake_inputs", True), fake_mode:
        torch_inputs = [input for input in sample_inputs if isinstance(input, torch.Tensor)]
    gm = aot_export_joint_simple(
        gm,
        torch_inputs,
        trace_joint=False,
        )
    return gm

#wrapping the backend with aot_autograd
def custom_backend_wrapped(backend):
    return aot_autograd(fw_compiler = backend)

def custom_backend_aot_autograd(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[Any], **kwargs: Any
) -> torch.nn.Module:
    custom_aot_autograd = custom_backend_wrapped(custom_backend)
    #tried this for the above error
    # fake_mode = detect_fake_mode(sample_inputs)
    # with unittest.mock.patch.object(fake_mode, "allow_non_fake_inputs", True), fake_mode:
    #     torch_inputs = [input.clone() for input in sample_inputs if isinstance(input, torch.Tensor)]
    # gm = custom_aot_autograd(gm, torch_inputs)
    gm = custom_aot_autograd(gm, sample_inputs)
    return gm
    

tp_model = torch.compile(
    tp_model,
    backend=custom_backend_aot_autograd,
    dynamic=False,
)
custom_backend_result = tp_model(inp)