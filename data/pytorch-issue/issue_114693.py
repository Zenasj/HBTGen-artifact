py
import torch
class MyMM(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.mm(b)
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad.mm(b.t()), a.t().mm(grad)
def my_mm(a, b):
    return MyMM.apply(a, b)
a = torch.randn([64, 64], device="cuda", dtype=torch.float32, requires_grad=True)
grad = a.clone()
print("torch", torch.__version__)
print("EAGER")
with torch.autocast("cuda"):
    out = my_mm(a, a)
out.backward(grad)
print("COMPILED")
my_mm_c = torch.compile(my_mm)
out = my_mm_c(a, a)
out.backward(grad)
"""
torch 2.2.0.dev20231128+cu118
EAGER
COMPILED
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [WARNING] speculate_subgraph: while introspecting the user-defined autograd.Function, we were unable to trace function `trampoline_autograd_fwd` into a single graph. This means that Dynamo was unable to prove safety for this API and will fall back to eager-mode PyTorch, which could lead to a slowdown.
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR] 'inline in skipfiles: MyMM.forward | decorate_fwd /home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/cuda/amp/autocast_mode.py, skipped according skipfiles.SKIP_DIRS'
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR] Traceback (most recent call last):
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/variables/higher_order_ops.py", line 240, in speculate_subgraph
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     output = f.call_function(tx, args, sub_kwargs)
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py", line 248, in call_function
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     return super().call_function(tx, args, kwargs)
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py", line 81, in call_function
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     return tx.inline_user_function_return(
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py", line 688, in inline_user_function_return
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py", line 2256, in inline_call
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     return cls.inline_call_(parent, func, args, kwargs)
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py", line 2371, in inline_call_
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     tracer.run()
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py", line 818, in run
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     and self.step()
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]         ^^^^^^^^^^^
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py", line 781, in step
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     getattr(self, inst.opname)(inst)
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py", line 470, in wrapper
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     return inner_fn(self, inst)
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]            ^^^^^^^^^^^^^^^^^^^^
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py", line 1252, in CALL_FUNCTION_EX
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     self.call_function(fn, argsvars.items, kwargsvars.items)
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py", line 652, in call_function
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     self.push(fn.call_function(self, args, kwargs))
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/variables/misc.py", line 643, in call_function
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     return self.obj.call_method(tx, self.name, args, kwargs)
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/variables/misc.py", line 504, in call_method
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     return tx.inline_call(tx, forward, args, kwargs)
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py", line 2256, in inline_call
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     return cls.inline_call_(parent, func, args, kwargs)
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py", line 2301, in inline_call_
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     result = InliningInstructionTranslator.check_inlineable(func)
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py", line 2280, in check_inlineable
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     unimplemented(
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]   File "/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_dynamo/exc.py", line 193, in unimplemented
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR]     raise Unsupported(msg)
[2023-11-28 11:40:57,863] [0/0] torch._dynamo.variables.higher_order_ops: [ERROR] torch._dynamo.exc.Unsupported: 'inline in skipfiles: MyMM.forward | decorate_fwd /home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/cuda/amp/autocast_mode.py, skipped according skipfiles.SKIP_DIRS'
/home/rzou/dev/nightly/env/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:140: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
"""