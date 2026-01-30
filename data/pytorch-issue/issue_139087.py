import torch

D=6144
hidden_states = torch.zeros([16384, 6144],           device="cuda:0", dtype=torch.bfloat16)
index         = torch.randint(0, 16384, (1, 32, 16384), device="cuda:0", dtype=torch.int64)
output        = torch.empty([1, 32, 16384, 6144],    device="cuda:0", dtype=torch.bfloat16)
hidden_states.index_add_(0, index.view(-1), output.view(-1, D))

D=6144
hidden_states = torch.zeros([16384, 6144],           device="cuda:0", dtype=torch.bfloat16)
index         = torch.randint(0, 16384, (1, 32, 16384), device="cuda:0", dtype=torch.int64)
output        = torch.empty([1, 32, 16384, 6144],    device="cuda:0", dtype=torch.bfloat16)
hidden_states.index_add_(0, index.view(-1), output.view(-1, D))