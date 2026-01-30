with self.fake_mode:
    result = super().run_node(n)

import random

import torch
import torch.nn as nn

import functorch

import torch._dynamo
from torch._dynamo.backends.common import aot_autograd
# from torch._dynamo.optimizations.training import aot_autograd as aot_autograd
# from torch._dynamo.utils import fake_mode_from_tensors
from torch._guards import detect_fake_mode

from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport

# PART 1 - define custom backend
use_fake = True # Change to False to not use fake mode and make infinite recursion disappear
class DeviceProp(torch.fx.Interpreter):
    def __init__(self, gm, fake_mode = None):
        super().__init__(gm)
        if fake_mode is None:
            fake_mode = torch._subclasses.FakeTensorMode()
        self.fake_mode = fake_mode

    def run_node(self, n):
        print("Node:", n, " Op:", n.op, "Target:", n.target)
        
        if not use_fake:
            result = super().run_node(n)
        else:
            with self._set_current_node(n):
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                new_args = []
                assert self.fake_mode
                for arg in args:
                    if isinstance(arg, torch.Tensor) and not isinstance(
                        arg, torch._subclasses.FakeTensor
                    ):
                        new_args.append(self.fake_mode.from_tensor(arg))
                    else:
                        new_args.append(arg)
            
                result = getattr(self, n.op)(n.target, new_args, kwargs)

        # Stuff related to device discovery
        # (...)
        
        return result

    def propagate(self, *args):
        return super().run(*args)

class MockOperatorSupport(OperatorSupport):
    def is_node_supported(self, submodules, node) -> bool:
        # Just mock for now, move all 'call_function' nodes into fused submodules.
        return node.op == "call_function"

def custom_compiler_inner(gm: torch.fx.GraphModule, example_inputs):
    print("####input FX graph:")
    gm.print_readable()
    
    # Take inputs' fake_mode and use it to trace through module and get actual devices placement
    fake_mode = detect_fake_mode(example_inputs)
    print("####FAKE_MODE: ", fake_mode)
    DeviceProp(gm, fake_mode).propagate(*example_inputs)
    
    # Use partitioner to partition out all necessary operations
    support_mock = MockOperatorSupport()
    partitioner = CapabilityBasedPartitioner(gm, support_mock, allows_single_node_partition=True)
    partitions = partitioner.propose_partitions()
    print("#### Partitions:")
    print(partitions)
    
    # Fuse these together again.
    fused_graph = partitioner.fuse_partitions(partitions)
    print("####fused FX graph:")
    fused_graph.print_readable()

    return functorch.compile.make_boxed_func(fused_graph.forward)

def custom_aot_backend(gm, example_inputs):
    # functorch.compile.config.use_functionalize = True
    # functorch.compile.config.use_fake_tensor = use_fake
    return aot_autograd(
        fw_compiler=custom_compiler_inner,
        bw_compiler=custom_compiler_inner,
    )(gm, example_inputs)

# PART 2 - define topology and iteration function.
num_classes = 10
learning_rate = 0.001

class MiniNetNoLinear(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 256, kernel_size=2, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Conv2d(256, num_classes, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu1(out)
        out = self.layer2(out)
        
        return out

model = MiniNetNoLinear(num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Change to SGD to make infinite recursion disappear
criterion = torch.nn.CrossEntropyLoss()

def iteration(x, y):
    optimizer.zero_grad()
    result = model(x)
    loss = result.sum().abs()
    loss.backward()
    optimizer.step()
    return loss, result

# PART 3 - compile and run

full_graph_compiled = torch.compile(iteration, backend=custom_aot_backend)

images = torch.rand(8, 1, 4, 4)
labels = torch.rand(8, 1)

for i in range(10):
    loss, _ = full_graph_compiled(images, labels)
    print("Iteration", i,":: Loss:", loss)