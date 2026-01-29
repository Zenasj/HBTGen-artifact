import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.split_module import split_module

# torch.rand(3, dtype=torch.float)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Original function with in-place operation
        def foo(x):
            x.add_(1)
            return None
        # Create FX graph for the function
        example_input = torch.randn(3)
        g = make_fx(foo, tracing_mode="fake")(example_input)
        
        # Split the graph using a dummy callback
        def split_callback(node):
            return 1  # Always split to partition 1
        self.split_graph = split_module(g, None, split_callback)
    
    def forward(self, x):
        # Forward pass simply returns input (split graph stored as submodule)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3)

