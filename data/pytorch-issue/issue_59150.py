import torch
print(torch.__version__)  # '1.8.1+cu111'
y = torch.tensor([[1, 2], [3, 4]])
torch.tile(y, (2, 2))  # this works
torch.tile(input=y, reps=(2, 2))  # TypeError: tile() missing 1 required positional arguments: "dims"
torch.tile(input=y, dims=(2, 2))  # this works, but should not