import torch
import torch.nn as nn

Python
# This will run your nn.Module or regular Python function with the example
# input that you provided. The returned callable can be used to re-execute
# all operations that happened during the example run, but it will no longer
# use the Python interpreter.
from torch.jit import trace
traced_model = trace(model, example_input=input)
traced_fn = trace(fn, example_input=input)

# The training loop doesn't change. Traced model behaves exactly like an
# nn.Module, except that you can't edit what it does or change its attributes.
# Think of it as a "frozen module".
for input, target in data_loader:
    loss = loss_fn(traced_model(input), target)