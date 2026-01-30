import torch

inner_indices = torch.tensor([[564,   0],
        [564,   0],
        [564,   0],
        [564,   1],
        [565,   1],
        [566,   0],
        [566,   0],
        [566,   0],
        [566,   0],
        [566,   0],
        [566,   0],
        [566,   1],
        [567,   0],
        [567,   0],
        [567,   0],
        [567,   0],
        [567,   0],
        [567,   1],
        [568,   0],
        [568,   0],
        [568,   0],
        [568,   1],
        [569,   0],
        [569,   0],
        [569,   0],
        [569,   1],
        [570,   0],
        [570,   0],
        [570,   0],
        [570,   0],
        [570,   0]])
data = torch.tensor([4, 4, 4, 4, 4, 4], dtype=torch.int32)
for u, v in zip(torch.tensor_split(data, inner_indices[:, 1]),
                torch.tensor_split(data, inner_indices[:, 1].contiguous())):
    if u.shape != v.shape:
        print(u, v)