# Patch set_rng_state as set_rng_state with fake tensors is
            # nonsensical. This does not affect the collection of metadata.

import torch

from typing import List
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func

torch._dynamo.config.capture_dynamic_output_shape_ops = True #Try dynamic shapes with torch.nonzero...
torch._dynamo.config.capture_scalar_outputs = True #Recommended to set this to True when setting capture_dynamic_output_shape_ops to True

def forward(x):
    batch_size = x.size()[0]
    molecule_size = x.size()[1]
    edges = torch.nonzero(x > 0.5, as_tuple=True)
    index_ij = ((edges[0] * molecule_size * molecule_size) + (edges[1] * molecule_size) + edges[2])
    vx = x.unsqueeze(2).repeat([1,1,8,1])
    vr_dist = torch.linalg.vector_norm((vx.unsqueeze(3) - x.unsqueeze(1).unsqueeze(2)), dim=-1)
    vr_x_ij = torch.abs(vr_dist.transpose(2,3).transpose(1,2).reshape(batch_size*molecule_size*molecule_size, 8).unsqueeze(-1).unsqueeze(-1)[index_ij])

    return vr_x_ij

graph_modules = []
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    graph_modules.append(gm)
    return make_boxed_func(gm.forward)

my_compiler = aot_autograd(fw_compiler=my_compiler,bw_compiler=my_compiler)

forward_aot = torch.compile(forward, fullgraph=True, dynamic=True, backend=my_compiler)

device = torch.device("cuda")

x = torch.rand([1, 5, 3], device=device)
x = x.requires_grad_(True)

vr_binned_x_ij = forward_aot(x)

import torch

from typing import List
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func

torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_scalar_outputs = True

def forward(x):
    batch_size = x.size()[0]
    molecule_size = x.size()[1]
    edges = torch.nonzero(x > 0.5, as_tuple=True)
    index_ij = ((edges[0] * molecule_size * molecule_size) + (edges[1] * molecule_size) + edges[2])
    dist_x = (x.unsqueeze(1) - x.unsqueeze(2)).sum(3)
    dist_indexed = dist_x[index_ij]
    return dist_indexed

graph_modules = []
def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    graph_modules.append(gm)
    return make_boxed_func(gm.forward)

my_compiler = aot_autograd(fw_compiler=my_compiler,bw_compiler=my_compiler)

forward_aot = torch.compile(forward, fullgraph=True, dynamic=True, backend=my_compiler)

device = torch.device("cuda")

x = torch.rand([1, 5, 3], device=device)
x = x.requires_grad_(True)

vr_binned_x_ij = forward_aot(x)