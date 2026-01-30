import torch
import numpy

from contextlib import AbstractContextManager

c = torch.compile
dev = 'cuda:0'


class expected(AbstractContextManager):
    def __init__(self, expected_exception_cls=None, subclass=False):
        self.expected = expected_exception_cls
        self.accept_subclass = subclass
    def __exit__(self, exc_type, exc_value, traceback):
        if self.expected is not None:
            assert exc_type is not None, 'Expected exception not raised'
        if issubclass(exc_type, self.expected) if self.accept_subclass else exc_type == self.expected:
            return True
        return False


def foo(a: numpy.ndarray, b: torch.Tensor):
    a = b.new_tensor(a)
    return torch.cat([a, b], dim=-1)

foo(
    numpy.array([ 1 ]),
    torch.randint(0, 10, [1], device=dev),
)

with expected(torch._dynamo.exc.TorchRuntimeError):
    c(foo)(
        numpy.array([ 1 ]),
        torch.randint(0, 10, [1], device=dev),
    )

with expected(RuntimeError):
    foo(
        torch.randint(0, 10, [1]),
        torch.randint(0, 10, [1], device=dev),
    )

with expected(torch._dynamo.exc.TorchRuntimeError):
    c(foo)(
        torch.randint(0, 10, [1]),
        torch.randint(0, 10, [1], device=dev),
    )