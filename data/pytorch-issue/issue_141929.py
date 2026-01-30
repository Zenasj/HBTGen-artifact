import torch

def optimized(*args, **kwargs):
    call_spec = runner.get_call_spec()
    in_spec = pytree.treespec_loads(call_spec[0])
    out_spec = pytree.treespec_loads(call_spec[1])
    flat_inputs = fx_pytree.tree_flatten_spec((args, kwargs), in_spec)
    flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
    flat_outputs = runner.run(flat_inputs)
    return pytree.tree_unflatten(flat_outputs, out_spec)