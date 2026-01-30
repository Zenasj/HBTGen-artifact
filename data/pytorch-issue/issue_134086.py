import torch
import torch.nn as nn
def fn(x):
  touched_outputs = []
  for v in x:
    y = v.__class__(torch.zeros(2, 2))
    touched_output = v * y
    touched_outputs.append(touched_output)
  return x.__class__(touched_outputs)
  
compiled_fn = torch.compile(fn, fullgraph=True)
data1 = torch.ones(2, 2)
data2 = torch.ones(2, 2)
output = compiled_fn((data1, data2))
print(output)

import torch
import torch.nn as nn

def fn(x):
  touched_outputs = []
  for v in x:
    y = v.__class__(torch.zeros(2, 2))
    touched_output = v * y
    touched_outputs.append(touched_output)
  return x.__class__(touched_outputs)
  
compiled_fn = torch.compile(fn, fullgraph=True)

data1 = torch.ones(2, 2)
data2 = torch.ones(2, 2)

output = compiled_fn([data1, data2])

print(output)

import torch
import torch.nn as nn
import collections

def fn(x):
  touched_outputs = []
  for v in x:
    y = v.__class__(torch.zeros(2, 2))
    touched_output = v * y
    touched_outputs.append(touched_output)
  return x.__class__(*touched_outputs)
  
compiled_fn = torch.compile(fn, fullgraph=True)

data1 = torch.ones(2, 2)
data2 = torch.ones(2, 2)

p = collections.namedtuple("Point", ["x", "y"])
t = p(data1, data2)

output = compiled_fn(t)

print(output)