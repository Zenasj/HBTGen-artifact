import torch

def fn():
    x = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
    y = [x[0], x[2], x[4]]
    return torch.LongTensor(y)