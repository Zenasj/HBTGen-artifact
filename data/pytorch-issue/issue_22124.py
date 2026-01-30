import torch.nn as nn

import torch

class ReLUDropoutInplace(torch.nn.Module):
    def __init__(self, p : float):
        super(ReLUDropoutInplace, self).__init__()
        self.p = p

    @torch.jit.script_method
    def forward(self, input):
        if self.training:
            p1m = 1. - self.p
            mask = torch.rand_like(input) < p1m
            mask *= (input > 0)
            return input.masked_fill_(~mask, 0).mul_(1. / p1m)
        else:
            return input.clamp_(min = 0)

m = ReLUDropoutInplace(0.5) # fails with:

#  File "foo.py", line 3, in <module>
#    class ReLUDropoutInplace(torch.nn.Module):
#  File "foo.py", line 8, in ReLUDropoutInplace
#    @torch.jit.script_method
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/__init__.py", line 1106, in script_method
#    ast = get_jit_def(fn, self_name="ScriptModule")
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 169, in get_jit_def
#    return build_def(ctx, py_ast.body[0], type_line, self_name)
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 209, in build_def
#    build_stmts(ctx, body))
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 125, in build_stmts
#    stmts = [build_stmt(ctx, s) for s in stmts]
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 125, in <listcomp>
#    stmts = [build_stmt(ctx, s) for s in stmts]
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 185, in __call__
#    return method(ctx, node)
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 343, in build_If
#    build_stmts(ctx, stmt.body),
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 125, in build_stmts
#    stmts = [build_stmt(ctx, s) for s in stmts]
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 125, in <listcomp>
#    stmts = [build_stmt(ctx, s) for s in stmts]
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 185, in __call__
#    return method(ctx, node)
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 288, in build_Return
#    return Return(r, None if stmt.value is None else build_expr(ctx, stmt.value))
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 185, in __call__
#    return method(ctx, node)
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 415, in build_Call
#    func = build_expr(ctx, expr.func)
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 185, in __call__
#    return method(ctx, node)
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 401, in build_Attribute
#    value = build_expr(ctx, expr.value)
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 185, in __call__
#    return method(ctx, node)
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 416, in build_Call
#    args = [build_expr(ctx, py_arg) for py_arg in expr.args]
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 416, in <listcomp>
#    args = [build_expr(ctx, py_arg) for py_arg in expr.args]
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 185, in __call__
#    return method(ctx, node)
#  File "/miniconda/lib/python3.7/site-packages/torch/jit/frontend.py", line 480, in build_UnaryOp
#    r = ctx.make_range(expr.lineno, expr.col_offset, expr.col_offset + len(op_token))
#TypeError: object of type 'NoneType' has no len()

torch.bitwise_or(o1, c1, out = o1) # works in torch.jit.script

o1 |= c1 # does not work in torch.jit.script