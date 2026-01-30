import torch

class Foo:
    foo: List[int] = [1, 2, 3]
    class Bar:
        bar: str = "1"
        class Baz:
            baz: int = 1

from torch._inductor import config as inductor_config


print(inductor_config.Foo)
print(repr(inductor_config.Foo.foo))
print(inductor_config.Foo.Bar)
print(repr(inductor_config.Foo.Bar.bar))
print(inductor_config.Foo.Bar.Baz)
print(repr(inductor_config.Foo.Bar.Baz.baz))