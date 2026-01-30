# Use `out` as something which not exist as an instance variable in the global scope
import torch
randint_result = torch.randint(0, 10, (100, 100), out = torch.empty(100, 100))

# Use `out` as something which exist as an instance variable in the global scope
import torch
out = torch.empty(100, 100)
randint_result = torch.randint(0, 10, (100, 100), out = out)
print(out)
## Output:
# tensor([[4., 2., 7.,  ..., 6., 4., 5.],
#         [5., 7., 6.,  ..., 9., 0., 0.],
#         [4., 7., 0.,  ..., 1., 5., 3.],
#         ...,
#         [4., 2., 3.,  ..., 6., 7., 8.],
#         [1., 7., 3.,  ..., 8., 4., 8.],
#         [6., 5., 9.,  ..., 2., 8., 2.]])

import inspect
import warnings

def verify_variable_names(func, local_vars, global_vars, check_vars = None):
    ## local vars
    sig = inspect.signature(func)
    local_names = list(sig.parameters.keys())
    local_values = [local_vars[name] for name in local_names]
    ## global vars & match
    external_var_names = {}
    for local_name, local_value in zip(local_names, local_values):
        if check_vars is not None and local_name not in check_vars:
            continue
        for global_name, global_value in global_vars.items():
            if id(global_value) == id(local_value):
                external_var_names[local_name] = global_name
                break
        if local_name not in external_var_names:
            warnings.warn(f"{local_name} in {func.__name__} not found as a valid variable in the global scope.") # if select to raise warning
            # raise RuntimeError(f"{local_name} in {func.__name__} not found as a valid variable in the global scope.") # if select to raise error
    # print(f"external_var_names: {external_var_names}")
def my_func(param1, param2):
    ## globals()[inspect.currentframe().f_code.co_name] refers to the current function call: my_func
    verify_variable_names(globals()[inspect.currentframe().f_code.co_name], locals(), globals())
    ## normal function body
    # ...

## example usage
a = 10
aa = 20.0 # as an interference term of `b` with the same value
b = 20.0
my_func(a, b) # success
my_func(a, 20.0) # raise warning/error

import builtins
from typing import (Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, overload, Sequence, Tuple, TypeVar, Union,)
import torch
from torch import contiguous_format, Generator, inf, memory_format, strided, SymInt, Tensor
from torch.types import (_bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size, Device, Number,)
from torch._prims_common import DeviceLikeType

import inspect
import warnings

def verify_variable_names(func, local_vars, global_vars, check_vars = None):
    ## local vars
    sig = inspect.signature(func)
    local_names = list(sig.parameters.keys())
    local_values = [local_vars[name] for name in local_names]
    ## global vars & match
    external_var_names = {}
    for local_name, local_value in zip(local_names, local_values):
        if check_vars is not None and local_name not in check_vars:
            continue
        for global_name, global_value in global_vars.items():
            if id(global_value) == id(local_value):
                external_var_names[local_name] = global_name
                break
        if local_name not in external_var_names:
            warnings.warn(f"{local_name} in {func.__name__} not found as a valid variable in the global scope.") # select to raise warning
            # raise RuntimeError(f"{local_name} in {func.__name__} not found as a valid variable in the global scope.") # select to raise error
    # print(f"external_var_names: {external_var_names}")

def randint(low: Union[_int, SymInt], high: Union[_int, SymInt], size: Sequence[Union[_int, SymInt]], *, out: Optional[Tensor] = None, dtype: Optional[_dtype] = None, layout: Optional[_layout] = None, device: Optional[Optional[DeviceLikeType]] = None, pin_memory: Optional[_bool] = False, requires_grad: Optional[_bool] = False) -> Tensor: 
    verify_variable_names(globals()[inspect.currentframe().f_code.co_name], locals(), globals(), check_vars = ['out'])
    return torch.randint(low, high, size, out = out, dtype = dtype, layout = layout, device = device, pin_memory = pin_memory, requires_grad = requires_grad)

out = torch.empty(100, 100)
randint_result1 = randint(0, 10, (100, 100), out = out) # Success
randint_result2 = randint(0, 10, (100, 100), out = torch.empty(100, 100)) # Raise warning

import torch
torch.randint(0, 10, (100, 100), out = torch.empty(100, 100))

import torch
torch.randint(0, 10, (100, 100), out = torch.empty(100, 100, dtype=torch.int64))