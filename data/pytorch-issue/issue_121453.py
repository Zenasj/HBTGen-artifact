import torch
import torch.nn as nn

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_iter, it, y):
        def cond_fn(out_iter, it, y):
            return it.sum() < 10

        def body_fn(out_iter, it, y):
            return (out_iter.clone(), it + y, y + 1)

        def outer_cond_fn(out_iter, it, y):
            return out_iter.sum() < 2

        def outer_body_fn(out_iter, it, y):
            out_iter, it, y = while_loop(cond_fn, body_fn, (out_iter, it, y))
            return (out_iter + 1, it, y)

        return while_loop(outer_cond_fn, outer_body_fn, (out_iter, it, y))

gm, _ = aot_export_module(M(), [torch.zeros(1), torch.zeros(1), torch.zeros(1)], trace_joint=False)