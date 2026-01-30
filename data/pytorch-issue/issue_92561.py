import torch
import torch.nn as nn

class Module(torch.nn.Module):
    def forward(self, pred, x):
        return self.indirection(pred, x)

    def indirection(self, pred, x):
        def true_fn(y):
            return y + 2

        def false_fn(y):
            return y - 2

        def shallow(x):
            return x * 2

        def deep(x):
            return cond(
                x[0][0] > 0,
                true_fn,
                false_fn,
                [x],
            )

        return cond(pred, shallow, deep, [x])