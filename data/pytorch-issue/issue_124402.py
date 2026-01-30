import torch

def isinstance_namedtuple(obj) -> bool:
    return (
            isinstance(obj, tuple) and
            hasattr(obj, '_asdict') and
            hasattr(obj, '_fields')
    )


@torch.compile(backend='eager', full_graph=True)
def f(tuple_of_tensors):
    assert isinstance(tuple_of_tensors, tuple)
    if isinstance_namedtuple(tuple_of_tensors):
        return type(tuple_of_tensors)(*(torch.mul(x, 2) for x in tuple_of_tensors))
    else:
        return type(tuple_of_tensors)([torch.mul(x, 2) for x in tuple_of_tensors])


from collections import namedtuple
MyTuple = namedtuple('MyNamedTuple', ['foo', 'bar'])

x = torch.ones(2)
y = torch.ones(2)

normal_tuple = (x, y)
my_tuple = MyTuple(foo=x, bar=y)

normal_out = f(normal_tuple)
named_out = f(my_tuple)

print(normal_out[0])
print(named_out.foo)