python
import torch
complex_tensor = [100 + 150j, 200 + 250j]

x = torch.tensor(complex_tensor, dtype=torch.complex32)
y = torch.tensor(complex_tensor, dtype=torch.complex64)
result = torch.abs(x)
result2 = torch.abs(y)
print(result)
print(result2)

tensor([180.2500, 320.2500], dtype=torch.float16)
tensor([180.2776, 320.1562])