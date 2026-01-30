import torch
x = torch.tensor([-1, -1, -1, -1])
print(torch.topk(x, 1)) # expected torch.return_types.topk(values=tensor([-1]), indices=tensor([3]))

# output from the function
# torch.return_types.topk(values=tensor([-1]), indices=tensor([2]))