py
import torch
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.experimental.proxy_tensor import make_fx
from torch._prims.context import TorchRefsMode
from copy import deepcopy

class PrimsOperatorSupport(torch.fx.passes.operator_support.OperatorSupport):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        return (
            node.op == "call_function"
            and str(node.target).startswith("prims.")
        )

def func(input):
    return torch.sin(input**2), torch.cos(input**2)

a = torch.randn(2, 4, 32, 16, dtype=torch.float32, device="cuda")

with TorchRefsMode():
    gm = make_fx(func)(a)

partitioned_gm = deepcopy(gm)
supported_ops = PrimsOperatorSupport()
partitioner = CapabilityBasedPartitioner(
    partitioned_gm, supported_ops
)
partitions = partitioner.propose_partitions()
partitioner.fuse_partitions(partitions)

print(f"len(partitions) = {len(partitions)}")
print(partitions)