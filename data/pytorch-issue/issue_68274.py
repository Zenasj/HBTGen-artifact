import torch

X = torch.randint(0, 23, (2, 3, 2, 2))
O = torch.permute(X, (2, 2, 1, 0))
# RuntimeError: repeated dim in permute

O = torch.permute(X, (2, 1, 0))
# RuntimeError: number of dims don't match in permute

O = torch.permute(X, (4, 3, 2, 1, 0))
# RuntimeError: number of dims don't match in permute

O = torch.permute(X, (3, 2, -1, 0))
# RuntimeError: repeated dim in permute

data2 = [0,1,2]
X2 = torch.tensor(data2)
O2 = torch.permute(X2, (0))
# permute(): argument 'dims' (position 2) must be tuple of ints, not int
# TypeError: permute(): argument 'dims' (position 2) must be tuple of ints, not int

O = torch.permute(X, (0, 1, 2, 3))
# do nothing since the dims doesn't change?