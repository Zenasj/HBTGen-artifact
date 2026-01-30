import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses.fake_tensor import FakeTensorMode

lib = torch.library.Library("test", "DEF")
impl_cpu = torch.library.Library("test", "IMPL", "CPU")
impl_meta = torch.library.Library("test", "IMPL", "Meta")

def foo_impl(x):
    return x + 1

lib.define("foo(Tensor a) -> Tensor")
impl_meta.impl("foo", foo_impl)
impl_cpu.impl("foo", foo_impl)


with enable_python_dispatcher():
    a = torch.ones(2, device='meta')
    print("@@@@@")
    b = torch.ops.test.foo.default(a)
    print(b)