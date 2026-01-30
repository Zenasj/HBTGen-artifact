import torch
import torch.nn as nn

network = nn.Linear(4, 4)
network = torch.jit.script(network)

path = "network.onnx"
arg = torch.randn(1, 4)

while True:
    if os.path.exists(path):
        os.remove(path)
    torch.onnx.export(network, arg, path)

    # debug tools, these don't affect the behaviour
    gc.collect()
    objgraph.show_growth()
    print([t.shape for t in objgraph.by_type("Tensor")[-4:]])
    print([t.storage().data_ptr() for t in objgraph.by_type("Tensor")[-4:]])
    print(gc.get_referrers(objgraph.by_type("Tensor")[-1]))