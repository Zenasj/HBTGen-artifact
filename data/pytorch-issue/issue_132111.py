import torch.nn as nn

import torch


tensor1 = torch.rand([1, 1, 4, 4], dtype=torch.float32)
tensor2 = torch.rand([1, 1, 4, 4], dtype=torch.float32)

nested_tensor = torch.nested.nested_tensor([tensor1, tensor2])
res = torch.nn.functional.adaptive_avg_pool2d(nested_tensor, 1)

...
nested_tensor = torch.nested.nested_tensor([tensor1, tensor2], layout=torch.jagged)
...