import torch
device = 'cpu'
size = 16
a = torch.full((size,), 6, dtype=torch.int32, device=device)
b = torch.full((size,), 3, dtype=torch.int32, device=device)
print(a & b)

tensor([    40960,         0,  46473216,     43691, 201326912,     43691,
                0,         0,     40960,         0,  46473216,     43691,
        201326912,     43691,         0,         0], dtype=torch.int32)

tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=torch.int32)