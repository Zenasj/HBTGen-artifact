import torch

HANDLED_FUNCTIONS = {}
class MyTensor(object):
    def __init__(self, torch_tensor):
         self.torch_tensor = torch_tensor
   
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, ScalarTensor))
            for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

import functools
def implements(torch_function):
    """Register a torch function override for ScalarTensor"""
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator

@implements(torch.rnn_tanh)
def rnn_tanh(input, hx, weights, bias, num_layers,
             dropout_p, train, bidirectional, batch_first):
    return my_computation(...)

model = RNN(10, 20, 2)
input = MyTensor(torch.randn(1, 5, 10))
model(input)

if typing.TYPE_CHECKING:
    assert ...

def mypy_assert(truthval):
    if typing.TYPE_CHECKING:
        assert truthval

mypy_assert(isinstance(x, torch.Tensor))

input_t: Tensor = input  # type: ignore

input_t = cast(Tensor, input)