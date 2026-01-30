import torch

cpu_tensor = torch.tensor([[0, 1], [2,1], [1, 0]])
torch.argmax(cpu_tensor, dim=1)  # tensor([1, 0, 0])

mps_tensor = cpu_tensor.to(torch.device('mps'))
torch.argmax(mps_tensor, dim=1)  # tensor([-9223372036854775808, -9223372036854775808, -9223372036854775808], device='mps:0')