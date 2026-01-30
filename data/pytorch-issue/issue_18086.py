import torch as t

y = t.tensor([[1, 1], [10, 10]])
mask = y.ge(5)
source = t.tensor([999])

y_clone = y.clone()

y.masked_scatter_(mask, source) # the intentional error.
y.equal(y_clone) # False. But it should remain the same.