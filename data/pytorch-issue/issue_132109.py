import torch
tensor1 = torch.rand([2, 3], dtype=torch.float64)
tensor1 = torch.nested.nested_tensor([tensor1]) # success if comment out this line
res = torch.transpose(tensor1, 0, 0)

...
nested_tensor = torch.nested.nested_tensor([tensor1], layout=torch.jagged)
nested_tensor.transpose(-1, -2)  # works, producing a non-contiguous nested tensor
...