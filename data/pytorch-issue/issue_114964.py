def _field_assign(frozen, name, value, self_name):
    # If we're a frozen class, then assign to our fields in __init__
    # via object.__setattr__.  Otherwise, just use a simple
    # assignment.
    #
    # self_name is what "self" is called in this function: don't
    # hard-code "self", since that might be a field name.
    if frozen:
        return f'__dataclass_builtins_object__.__setattr__({self_name},{name!r},{value})'
    return f'{self_name}.{name}={value}'

import dataclasses

import torch


@dataclasses.dataclass
class A:
    a: int


def fn() -> None:
    a = A(1)
    object.__setattr__(a, "a", 2)


comp_out = torch._dynamo.optimize(nopython=True)(fn)()