import torch
import torch.nn as nn

class ModuleClosureReproError(torch.nn.Module):
            # error
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, pred, x):
                y = x + x

                def true_fn(val):
                    return self.linear(val) * (x + y)

                def false_fn(val):
                    return val * (y - x)

                return cond(pred, true_fn, false_fn, [x])