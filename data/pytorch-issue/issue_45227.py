import torch

graph = torch.fx.Graph()
x = torch.fx.Proxy(graph.placeholder('x'))
relu = torch.relu(x)

with torch.fx.graph.insert_before(relu.node):
    y = torch.neg(x)
    z = torch.tanh(y)

graph.output((relu.node, z.node))