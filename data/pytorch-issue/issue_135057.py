py
import torch
from torch import _inductor as inductor

@torch.library.custom_op("mylib::sin", mutates_args={}) # E: Untyped decorator makes f
def sin(x: torch.Tensor) -> torch.Tensor:
    return x.sin()

x = torch.randn(3, requires_grad=True)
y = sin(x.clone()).sum()

from torch._dynamo.compiled_autograd import enable


def make_compiler_fn(fullgraph=True, dynamic=True): # E: Function is missing a type an
    def _compiler_fn(gm): # E: Function is missing a type annotation  [no-untyped-def]
        """Same as torch.compile() but counts number of compiles"""

        def _inner_compiler(gm_, example_inputs_): # E: Function is missing a type ann
            return inductor.compile(gm_, example_inputs_)

        return torch.compile(
            gm, backend=_inner_compiler, fullgraph=fullgraph, dynamic=dynamic
        )

    return _compiler_fn


compiler_fn = make_compiler_fn()


with enable(compiler_fn):
    y.backward()