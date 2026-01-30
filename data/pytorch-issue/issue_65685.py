import torch

x_tensor = torch.ones((3, ), dtype=torch.int8)
y_tensor = torch.tensor(1, dtype=torch.float64)
out = torch.mul(x_tensor, y_tensor) # torch.float64

x_tensor = torch.ones((3, ), dtype=torch.int8)
y_tensor = torch.tensor(1, dtype=torch.int64)
out = torch.mul(x_tensor, y_tensor) # torch.int8

x_tensor = torch.ones((3, 3), dtype=torch.int8)
y_tensor = torch.ones((3, 3), dtype=torch.int64)
out = torch.mul(x_tensor, y_tensor) # torch.int64

x_tensor = torch.ones((3, 3), dtype=torch.int8)
out = torch.mul(x_tensor, 9223372036854775807) # torch.int8