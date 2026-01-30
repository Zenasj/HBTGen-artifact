import torch
from torch._subclasses import FakeTensorMode
import torch.fx as fx

def graph_call_function(graph, fn, *args, **kwargs):
    fake_args, fake_kwargs = torch.utils._pytree.tree_map(
        lambda node: node.meta["val"] if isinstance(node, fx.Node) else node,
        (args, kwargs),
    )
    with FakeTensorMode() as fake_mode:
        fake_result = fn(*fake_args, **fake_kwargs)
    node = graph.call_function(fn, args, kwargs)
    node.meta["val"] = fake_result
    return node

# Create fake tensors and FX graph with nodes containing them in metadata
fake_mode = FakeTensorMode()
real_tensor = torch.rand(4)
fake_tensor = fake_mode.from_tensor(real_tensor)

graph = fx.Graph()
placeholder_node = graph.placeholder('x')
placeholder_node.meta["val"] = fake_tensor

# Create a node that stores FakeTensor in its metadata
node = graph_call_function(graph, torch.add, placeholder_node, placeholder_node)

import torch
from torch._subclasses import FakeTensorMode
import torch.fx as fx


fake_mode = FakeTensorMode()

def graph_call_function(graph, fn, *args, **kwargs):
    fake_args, fake_kwargs = torch.utils._pytree.tree_map(
        lambda node: node.meta["val"] if isinstance(node, fx.Node) else node,
        (args, kwargs),
    )
    global fake_mode
    with fake_mode:
        fake_result = fn(*fake_args, **fake_kwargs)
    node = graph.call_function(fn, args, kwargs)
    node.meta["val"] = fake_result
    return node

# Create fake tensors and FX graph with nodes containing them in metadata
real_tensor = torch.rand(4)
fake_tensor = fake_mode.from_tensor(real_tensor)

graph = fx.Graph()
placeholder_node = graph.placeholder('x')
placeholder_node.meta["val"] = fake_tensor

# Create a node that stores FakeTensor in its metadata
node = graph_call_function(graph, torch.add, placeholder_node, placeholder_node)