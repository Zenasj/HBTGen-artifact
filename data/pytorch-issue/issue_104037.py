# torch.rand(1, 128, 384), torch.randint(128, (1, 128))  # Input shapes
import torch
from torch import Tensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions

def embedding_dense_backward_op(grad_output: Tensor, index: Tensor) -> Tensor:
    return torch.ops.aten.embedding_dense_backward(
        grad_output, index, 512, 1, True
    )

def strip_overloads(gm):
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Generate FX graph with decomposition and strip overloads
        grad_output = torch.randn(1, 128, 384)
        index = torch.randint(high=128, size=(1, 128))
        fx_g = make_fx(
            embedding_dense_backward_op,
            decomposition_table=get_decompositions([torch.ops.aten.embedding_dense_backward]),
        )(grad_output, index)
        strip_overloads(fx_g)
        self.fx_graph = fx_g

    def forward(self, inputs):
        return self.fx_graph(*inputs)

def my_model_function():
    return MyModel()

def GetInput():
    grad_output = torch.randn(1, 128, 384)
    index = torch.randint(high=128, size=(1, 128))
    return (grad_output, index)

