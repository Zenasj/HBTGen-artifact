import torch
a = torch.eye(1024, 1024, device=torch.device("cuda")).reshape(1, 1, 1024, 1024)
a[..., -1, -1] = 0
print(torch.linalg.matrix_rank(a))

torch.linalg.svdvals(a)[..., 0] * 1024 * torch.finfo(torch.float32).eps