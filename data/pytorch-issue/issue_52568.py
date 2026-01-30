import torch

torch.manual_seed(0)
generator = torch.Generator()
seed = int(torch.empty((), dtype=torch.int64).random_().item())
# seed = 0
generator.manual_seed(seed)
print(torch.randperm(10, generator=None).tolist())
#[3, 7, 5, 2, 0, 8, 1, 6, 9, 4]

torch.manual_seed(0)
generator = torch.Generator()
# seed = int(torch.empty((), dtype=torch.int64).random_().item())
seed = 0
generator.manual_seed(seed)
print(torch.randperm(10, generator=None).tolist())
#[4, 1, 7, 5, 3, 9, 0, 8, 6, 2]