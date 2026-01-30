import torch

class TestTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        with torch._C.DisableTorchFunctionSubclass():
            out = func(*args, **kwargs)
            if func == torch.clone:
                return out * 2
            return out

def fn(x):
    return torch.clone(x)

with torch._dynamo.config.patch(traceable_tensor_subclasses={TestTensor}):
    inp = torch.ones(4, 4)
    x = TestTensor(inp)
    torch._dynamo.mark_dynamic(x, 0)
    compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
    out = compiled_fn(x)
    assert torch.allclose(out, torch.ones(4, 4) * 2)