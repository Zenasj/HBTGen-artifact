import torch.nn as nn

import torch

def test_lazy_module():

    class ModWithOneLazyLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.LazyLinear(8)

        def forward(self, x):
            return self.layer(x)

    # This allows us to restart tracing without clearing speculation log
    def id_and_graph_break(x):
        torch._dynamo.graph_break()
        return x

    @torch.compile()
    def foo(mod, x):
        # We'll trace this `mod(x)` call 2 times (because of the manual graph
        # break right after it)
        #
        # - 1st time the `mod.layer` would be treated as an `NNModuleVariable`
        # (because it's a `LazyLinear`), so we won't trace into its call method,
        # and would leave it as an opaque fx call to the linear layer.
        #
        # NOTE that as part of tracing `mod.layer(x)` we'd hit
        # `initialize_lazy_module`, which changes the underlying `mod.layer`
        # from a LazyLinear to a Linear instance.
        #
        # - 2nd time the `mod.layer` would be treated as an
        # `UnspecializedNNModuleVariable` (because it's now a `Linear`), and
        # we'd trace into the linear layer rather than keeping it as a single fx
        # call. This breaks speculation log.
        res = mod(x)
        res2 = id_and_graph_break(res)
        return res

    x = torch.ones(10, 3)
    mod = ModWithOneLazyLinear()
    foo(mod, x)

test_lazy_module()