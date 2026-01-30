import torch
from torch import Tensor
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions


def embedding_dense_backward_op(grad_output, index):
    return torch.ops.aten.embedding_dense_backward(
        grad_output, index, 512, 1, True
    )


grad_output = torch.randn(1, 128, 384)
index = torch.randint(high=128, size=(1, 128))

fx_g = make_fx(
    embedding_dense_backward_op,
    decomposition_table=get_decompositions(
        [
            torch.ops.aten.embedding_dense_backward,
        ]
    ),
)(grad_output, index)

fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
fx_g.recompile()


def strip_overloads(gm):
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.
    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()


strip_overloads(fx_g)

ts_graph = torch.jit.script(fx_g)