import torch.nn as nn

import torch
from torch._functorch.aot_autograd import aot_export_joint_simple

class Aliased(torch.nn.Module):
    def forward(self, x):
        return (x[None],)

gm = aot_export_joint_simple(Aliased(), [torch.rand((5, 5))], trace_joint=False)

def aot_inference_module(
    func: Callable,
    args,
    *,
    decompositions: Optional[Dict] = None,
    dynamic: bool = False,
) -> torch.fx.GraphModule:
    """
    A utility built for inference, tracing only the forward graph and retaining module attributes.

    The function has both export and compile-inspired components. It can support dynamic shapes/dimensions.

    From `aot_export_joint_simple` it has:
    - This function enforces the `no_grad` context, being an inference-only method
    - This function does not hoist the parameters or buffers to inputs, instead leaving them as `get_attr` nodes
    - `args` cannot contain any pytrees (they must have been pytree_flattened already)

    From `aot_module_simplified` it has:
    - This function does not check or enforce any restrictions on input/output aliasing or input mutation
      - This could be controlled by an argument such as `allow_input_aliasing_mutation: bool = False`, if desired
    """