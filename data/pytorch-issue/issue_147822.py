# torch.rand(4, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.fx as fx
from torch._subclasses import FakeTensorMode

fake_mode = FakeTensorMode()  # Global fake_mode instance

def graph_call_function(graph, fn, *args, **kwargs):
    fake_args, fake_kwargs = torch.utils._pytree.tree_map(
        lambda node: node.meta["val"] if isinstance(node, fx.Node) else node,
        (args, kwargs),
    )
    with fake_mode:
        fake_result = fn(*fake_args, **fake_kwargs)
    node = graph.call_function(fn, args, kwargs)
    node.meta["val"] = fake_result
    return node

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        real_tensor = torch.rand(4)
        fake_tensor = fake_mode.from_tensor(real_tensor)
        
        graph = fx.Graph()
        placeholder_node = graph.placeholder('x')
        placeholder_node.meta["val"] = fake_tensor
        
        # Create the add node using the corrected graph_call_function
        node = graph_call_function(graph, torch.add, placeholder_node, placeholder_node)
        graph.output(node)
        
        # Create an empty root module to attach the graph
        class EmptyModule(torch.nn.Module):
            pass
        
        self.graph_module = torch.fx.GraphModule(EmptyModule(), graph)
    
    def forward(self, x):
        return self.graph_module(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4)

