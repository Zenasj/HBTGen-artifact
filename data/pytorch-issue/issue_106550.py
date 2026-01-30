import torch

torch._dynamo.config.suppress_errors=False

@torch.compile(backend="eager")
def test():
    args = (0, 1, 1, 0)
    _, arg_spec = torch.utils._pytree.tree_flatten(args)

    # Dynamo doesn't know how to handle this.
    torch.utils._pytree._broadcast_to_and_flatten(0, arg_spec)
    
test()