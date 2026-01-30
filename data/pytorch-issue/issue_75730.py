import torch

# Create a multi-dimensional tensor, with a single zero-length dimension.
input_tensor = torch.randint(3, [0, 4])

# Call `torch.unique` on the tensor, along the zero-length dimension.
input_tensor.unique(dim=0)

# Output:
# torch.tensor([], dtype=torch.int64)

torch.randint(3, [0, 4]).unique(dim=0)

# Output:
# torch.tensor([], size=(0, 4), dtype=torch.int64)




# Or, for a differently-sized input tensor:
torch.randint(3, [2, 0, 9]).unique(dim=1)

# Output:
# torch.tensor([], size=(2, 0, 9), dtype=torch.int64)

torch.randint(3, [0, 4]).unique(dim=0)

# Output:
# torch.tensor([], dtype=torch.int64)




torch.randint(3, [2, 0, 9]).unique(dim=1)

# Output:
# torch.tensor([], dtype=torch.int64)

import torch

# Create a multi-dimensional tensor, with a single zero-length dimension.
input_tensor = torch.randint(3, [0, 4])

# Call `torch.sort` on the tensor, along the zero-length dimension.
input_tensor.sort(dim=0).values

# Output:
# torch.tensor([], size=(0, 4), dtype=torch.int64)

import torch

# Create a multi-dimensional tensor, with a single zero-length dimension.
input_tensor = torch.randint(3, [0, 4])

# Call `torch.sort` on the tensor, along the zero-length dimension.
torch.cat([input_tensor]*2, dim=0)

# Output:
# torch.tensor([], size=(0, 4), dtype=torch.int64)