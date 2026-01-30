import torch
import random

py
from torch._dynamo import trace_rules
import numpy as np


def new_numpy_function_ids():
    unsupported_funcs = {"seed", "ranf", "get_bit_generator", "RandomState", "set_bit_generator", "sample"}

    def is_supported(k, v, mod):
        if not callable(v):
            return False
        if not getattr(v, "__module__", None):
            return True
        if v.__module__ == mod.__name__:
            return True
        if v.__module__ == "numpy.random.mtrand" and mod.__name__== "numpy.random" and k not in unsupported_funcs:
            return True
        return False
    rv = {}
    for mod in trace_rules.NP_SUPPORTED_MODULES:
        for k, v in mod.__dict__.items():
            if is_supported(k, v, mod):
                rv[id(v)] = f"{mod.__name__}.{k}"
    return rv

def old_numpy_function_ids():
    rv = {}
    for mod in trace_rules.NP_SUPPORTED_MODULES:
        rv.update(
            {
                id(v): f"{mod.__name__}.{k}"
                for k, v in mod.__dict__.items()
                if callable(v)
                and (getattr(v, "__module__", None) or mod.__name__) == mod.__name__
            }
        )
    return rv

rv1 = set(old_numpy_function_ids().values())
rv2 = set(new_numpy_function_ids().values())

for v in (rv1 - rv2):
    print(v)
print("****")
for v in (rv2 - rv1):
    print(v)