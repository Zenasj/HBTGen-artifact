import torch
import torch._dynamo as dynamo

dynamo_callable = dynamo.optimize(nopython=True)(lambda x: bool(x))
dynamo_callable(torch.tensor(1, dtype=torch.bool))

dynamo_callable = dynamo.optimize(nopython=True)(lambda: torch.ops.aten.Bool(5))
dynamo_callable()