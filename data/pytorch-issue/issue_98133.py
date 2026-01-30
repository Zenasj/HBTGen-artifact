import torch


class Foo:
    __cuda_array_interface__ = {
        "data": (0, False),
        "typestr": "|i",
        "shape": (0, ),
    }


foo = Foo()
t = torch.asarray(foo, device="cuda")
assert t.is_cuda, t.device
print(t)