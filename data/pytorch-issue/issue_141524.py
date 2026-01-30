import torch
import torch.nn as nn

class M(torch.nn.Module):
            def forward(self, x, y, z):
                a = y.shape[0]
                b = z.shape[0]

                def true_fn(x):
                    return x + a

                def false_fn(x):
                    return x + b * z

                # When exporting with non-strict: a and b are symints,
                # so torch.compile need to wrap and trace symint inputs.
                return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

def f(x):
  s0 = x.shape[0]

  def true_fn(x):
     if isinstance(s0, int):
       return x + s0
     else:
       return x - s0

  torch.cond(s0 > 4, true_fn, ...)