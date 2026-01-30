import gc
import weakref

import torch

def test_static_address_finalizer():
    def inner(y):
        y

    inner = torch._dynamo.optimize("eager")(inner)

    p_ref = None

    x = torch.randn((10, 10), device="cuda:0")
    inner(x)

    p_ref = weakref.ref(x)
    assert p_ref() is not None
    del x
    gc.collect()
    assert p_ref() is None
    print("Success!")

test_static_address_finalizer()