def f(x):
    return x * 2

import torch
gm = torch.fx.symbolic_trace(f)

wrapped_forward = torch._dynamo.disable(gm.forward)
got_inner_forward = torch._dynamo.eval_frame.innermost_fn(wrapped_forward)

print(hasattr(got_inner_forward, '__self__')) # True

lazy_gm = torch.fx._lazy_graph_module._LazyGraphModule.from_graphmodule(gm)
wrapped_lazy_forward = torch._dynamo.disable(lazy_gm.forward)
got_lazy_inner_forward = torch._dynamo.eval_frame.innermost_fn(wrapped_lazy_forward)

print(hasattr(got_lazy_inner_forward, '__self__')) # False