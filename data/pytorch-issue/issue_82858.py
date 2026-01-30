import torch.nn as nn
import torchvision

from torchvision.models import resnet18
import torch.utils._pytree as pytree
from torch.nn.utils.stateless import functional_call
from functorch.compile import ts_compile
from functorch.compile import default_decompositions
import torch.utils._pytree as pytree
from torchdynamo.testing import reduce_to_scalar_loss


def trace_model(model, inputs):
    """
    Get the full graph (both forward and backward) of `model` on `inputs`
    The moddel should have a single forward and a single backward graph
    """
    def f(params, inp):
        out = functional_call(model, params, inp)
        loss = reduce_to_scalar_loss(out)
        loss.backward()
        return [param.grad for param in params.values()]
    
    params = dict(model.named_parameters())
    traced_graph = make_fx(f, decomposition_table=default_decompositions)(params, inputs)
    return traced_graph, params

inputs = torch.rand(1, 3, 224, 224, device="cuda")
model = resnet18(pretrained=True)
model.to("cuda")


model.eval()
traced_graph, params = trace_model(model, inputs)
traced_graph.graph.set_codegen(torch.fx.graph.CodeGen())  # avoid recursive pytree
arg_list, spec  = pytree.tree_flatten([params, inputs])
# print(traced_graph)
script_f = ts_compile(traced_graph, 0)
with torch.no_grad():
    script_f(*arg_list)
    script_f(*arg_list)