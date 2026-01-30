import torch
import torch.nn as nn

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("buffer", torch.randn(4, 4))

    def forward(self, x):
        def true_fn(x):
            self.buffer.add_(5)
            return x.cos() + self.buffer.sum()

        def false_fn(x):
            return x.sin()

        a = torch.cond(x.shape[0] > 4, true_fn, false_fn, [x])
        return (a + 3, a + 4)


inp = torch.randn(3, 4)
ep = torch.export.export(M(), (inp,))
print(ep.graph_module.graph)
print(ep.graph_module.true_graph_0.graph)