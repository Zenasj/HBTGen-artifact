import torch 

x = torch.randn(32, 3)
results = []
for xi in x:
  y = torch.triu(xi)
  results.append(y)
"""
triu: input tensor must have at least 2 dimensions
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-7-d726203efb0e> in <module>
      4 results = []
      5 for xi in x:
----> 6   y = torch.triu(xi)
      7   results.append(y)
RuntimeError: triu: input tensor must have at least 2 dimensions
"""