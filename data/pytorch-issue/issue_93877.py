import torch
mps_index = torch.tensor([], device='mps', dtype=torch.int32)
mps_tensor = torch.tensor([1, 2, 3, 4, 5], device='mps')
print(torch.index_select(mps_tensor, 0, mps_index))

import torch
cpu_index = torch.tensor([], dtype=torch.int32)
cpu_tensor = torch.tensor([1, 2, 3, 4, 5])
print(torch.index_select(cpu_tensor, 0, cpu_index))

tensor([], dtype=torch.int64)