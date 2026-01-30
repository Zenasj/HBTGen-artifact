getattr(node.graph.owning_module, node.target)

node_layer = node.target.split(".")
module = node.graph.owning_module
for layer in node_layer:
    module = getattr(module, layer)