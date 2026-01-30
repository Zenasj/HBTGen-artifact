import torch
import torch.nn as nn

model = nn.Linear(1, 1)
x = torch.randn(1)

print(x.shape)
# torch.Size([1])

# works
out = model(x)
print(out.shape)
# torch.Size([1])

# fails
x = torch.randn(1).squeeze()
print(x.shape)
# torch.Size([])

out = model(x)
# RuntimeError: ArrayRef: invalid index Index = 18446744073709551615; Length = 0