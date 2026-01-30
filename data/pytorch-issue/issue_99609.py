import torch
from torch import _dynamo as dynamo


def func():
    a = {"str": torch.rand([2, 2, 3])}
    t2 = torch.tensor([0, 1], dtype=torch.int64)

    def inner(t):
        for k, v in a.items():
            a[k] = v[t]

    inner(t2 < 1)


ex = dynamo.explain(func)[-1]
print(ex)

torch._logging.set_logs(dynamo=logging.DEBUG)

dynamo.config.dynamic_shapes = True
dynamo.config.capture_dynamic_output_shape_ops = True