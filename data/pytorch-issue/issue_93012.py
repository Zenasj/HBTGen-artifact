import torch

@torch.jit._drop
def __fx_create_arg__(self, tracer: torch.fx.Tracer) -> torch.fx.node.Argument:
    # torch.fx classes are not scriptable
    return tracer.create_node(
        "call_function",
        CFX,
        args=(tracer.create_arg(self.features),),
        kwargs={},
    )

def __iter__(self) -> Iterator[torch.Tensor]:
    return iter(self.a)