import torch
import torch.nn as nn

@torch.jit.interface
class TestInterface(torch.nn.Module):
    def forward(self, x: int) -> int:
        r""" Any doc-string """
        pass

Module(body=[FunctionDef(name='forward', args=arguments(posonlyargs=[], args=[arg(arg='self', annotation=None, type_comment=None), arg(arg='x', annotation=Name(id='int', ctx=Load()), type_comment=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=[Expr(value=Constant(value=' Any doc-string ', kind=None)), Pass()], decorator_list=[], returns=Name(id='int', ctx=Load()), type_comment=None)], type_ignores=[])