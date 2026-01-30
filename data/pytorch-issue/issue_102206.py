import torch.nn as nn

import torch
from torch._functorch.aot_autograd import aot_export_joint_simple
    
def f(x, weight):
    def cb(grad_out):
        y = grad_out.sin()  # stand-in for collective result
        weight.data.copy_(y)
        # Better would be:
        #   weight.data = y
        return grad_out
    r = x @ weight
    r.register_hook(cb) 
    return (r,)

gm = aot_export_joint_simple(
    f,
    [torch.randn(2, 3, requires_grad=True), torch.randn(3, 4)],
    trace_joint=True,
)
gm.print_readable()

import torch
from torch._functorch.aot_autograd import aot_export_joint_simple
    
def f(x, weight):
    def cb(grad_out):
        y = grad_out.sin()  # stand-in for collective result
        weight.data.copy_(y)
        # Better would be:
        #   weight.data = y
        return grad_out
    r = x @ weight
    r.register_hook(cb) 
    return (r,)

gm = aot_export_joint_simple(
    f,
    [torch.randn(2, 2, requires_grad=True), torch.randn(2, 2)],
    trace_joint=True,
)
gm.print_readable()

class joint_helper(torch.nn.Module):
    def forward(self, primals, tangents):
        primals_1: "f32[2, 2]"; primals_2: "f32[2, 2]"; tangents_1: "f32[2, 2]"; 
    
        primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
        # No stacktrace found for following nodes
        mm: "f32[2, 2]" = torch.ops.aten.mm.default(primals_1, primals_2);  primals_1 = None
        sin: "f32[2, 2]" = torch.ops.aten.sin.default(tangents_1)
        detach: "f32[2, 2]" = torch.ops.aten.detach.default(primals_2)
        detach_1: "f32[2, 2]" = torch.ops.aten.detach.default(detach);  detach = None
        copy: "f32[2, 2]" = torch.ops.aten.copy.default(detach_1, sin);  detach_1 = sin = None
        t_1: "f32[2, 2]" = torch.ops.aten.t.default(copy)
        mm_1: "f32[2, 2]" = torch.ops.aten.mm.default(tangents_1, t_1);  tangents_1 = t_1 = None
        copy_: "f32[2, 2]" = torch.ops.aten.copy_.default(primals_2, copy);  primals_2 = copy = None
        return pytree.tree_unflatten([mm, mm_1, None], self._out_spec)