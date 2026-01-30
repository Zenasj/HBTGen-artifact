import torch

def fn(x):
    x = x + 1
    try:
        raise AssertionError()
    except AssertionError:
        pass
    x = x + 2
    return x

x = torch.zeros(3)
eager = fn(x)

from torch._dynamo.testing import CompileCounter
cnt = CompileCounter()
compiled_fn = torch._dynamo.optimize(backend=cnt)(fn)
compiled = compiled_fn(x)

torch.testing.assert_close(eager, compiled)
assert cnt.frame_count == 2, cnt.frame_count  # 1, but should be 2!