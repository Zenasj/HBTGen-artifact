import torch

g, p, o = _model_to_graph(module, torch.ones(1, 10))
for n in g.nodes():
    for v in n.outputs():
        print(v.debugName())