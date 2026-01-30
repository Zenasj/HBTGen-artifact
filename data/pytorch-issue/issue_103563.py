import torch.nn as nn

VariableTracker

VariableTracker.recursively_contains

import torch
import logging
import torch._dynamo
import torch._inductor

def f(x):
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    for i in range(3):
        if i == 0:
            data[0][i] = x
        else:
            data[0][i] = data[0][i - 1] + 1
    return data[0][-1]

x = torch.rand(2)

print("=== Eager ===")
print(f(x))

opt_f = torch.compile(backend="eager")(f)
print("=== Compile ===")
print(opt_f(x))

from typing import Dict
import torch

class Configs:
    def __init__(self, channels):
        self.channels = channels
class Module2(torch.nn.Module):
    def __init__(self, val):
        super().__init__()
        self.model_config = Configs(val)
    ####### PROBLEMATIC METHOD #######
    def flatten_nested_dict(self, model_config) -> Dict[str, Dict[str, str]]:
        assert(model_config.channels)
        embedding_sequence_groups = {}
        for (event_type, d) in model_config.channels.items():
            embedding_sequence_groups[event_type] = {}
            for entity_type, l in d.items():
                embedding_sequence_groups[event_type][entity_type] = l[0]
        return embedding_sequence_groups

    def wrapped_nested_dict(self, foo):
        return self.flatten_nested_dict(self.model_config)
    def forward(self, foo):
        retval = self.wrapped_nested_dict(foo)
        print(retval)
        return foo + len(retval)

m = Module2({
    "a": {"aa": ["foo1", "foo12"], "ab": ["foo2", "foo22"]},
    "b": {"ba": ["foo3", "foo32"]},
    "c": {"ca": ["foo4"]},
})
compiled = torch.compile(m)

print(m(torch.tensor([1], dtype=torch.double, requires_grad=True)))
print(compiled(torch.tensor([1], dtype=torch.double, requires_grad=True)))