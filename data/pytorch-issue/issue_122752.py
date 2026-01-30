import torch
import torch.export
import torch.nn as nn
from torch._decomp import get_decompositions


LIBRARY = torch.library.Library("torch_mlir_test", "DEF")
# We do not need to define a meta op in this case because the functionalizer
# figures it out (no returns). The result is the same if we do define one.
LIBRARY.define("inplace_modify(Tensor(a!) x) -> ()")


# Specific decompositions don't matter. It merely seems to be required to
# use run_decompositions with an op defined in this way.
decomposition_table = get_decompositions([
    torch.ops.aten.addmm
])

class Basic(nn.Module):
    def forward(self, x):
        torch.ops.torch_mlir_test.inplace_modify(x)
        return x * x

ep = torch.export.export(Basic(), (torch.randn(3, 4),))
ep.run_decompositions(decomposition_table)

def _patch_op_dispatch(op):
    if torch.__version__ >= "2.3.0" and torch.__version__ < "2.4":
        # Around the torch 2.3.0 release cut, there was a regression such that
        # running decompositions in a functionalized context did not work
        # with Python registered ops. The issue is that they have an incomplete
        # list of mode handler registrations and cannot handle the
        # FunctionalTensorMode. Since we only have a handful of these, and
        # since we can assume that for the sake of expediency, functional
        # dispatch is basically the same as fake tensor dispatch, we just
        # take the fake tensor registration and dup it onto the functional
        # registration.
        # Note that the torch._higher_order_ops.auto_functionalize is registered
        # in Python and is itself broken, it needs to be monkey patched.
        # See: https://github.com/pytorch/pytorch/issues/122752
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch._subclasses.functional_tensor import FunctionalTensorMode

        t = op.python_key_mode_table
        if FunctionalTensorMode not in t:
            handler = t[FakeTensorMode]
            t[FunctionalTensorMode] = handler


_patched_op_dispatch_for_export = False


def _patch_op_dispatch_for_export():
    global _patched_op_dispatch_for_export
    if _patched_op_dispatch_for_export:
        return
    _patched_op_dispatch_for_export = True
    import torch._higher_order_ops.auto_functionalize

    _patch_op_dispatch(torch._higher_order_ops.auto_functionalize.auto_functionalized)