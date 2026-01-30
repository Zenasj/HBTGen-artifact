import torch

def foo(slice_31: "f32[2, 1, 8192, 1]"):
    constant_pad_nd: "f32[2, 1, 8192, 8]" = torch.ops.aten.constant_pad_nd.default(slice_31, [0, 7], 0.0);  slice_31 = None
    slice_32: "f32[2, 1, 8192, 1]" = torch.ops.aten.slice.Tensor(constant_pad_nd, -1, 0, 1);  constant_pad_nd = None
    expand_6: "f32[2, 32, 8192, 8192]" = torch.ops.aten.expand.default(slice_32, [2, 32, 8192, 8192])

    return expand_6, expand_6.clone()
 
from torch._dynamo.debug_utils import aot_graph_input_parser
kwargs = aot_graph_input_parser(foo)

torch.compile(foo)(*kwargs.values())