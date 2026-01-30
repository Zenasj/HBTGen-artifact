import torch
device = torch.device('cpu')

# replace [1, 10] with [10] to make it work in torch.compile and fail in eager
x = torch.zeros([1, 10], device=device)
xs = torch.ones([2, 10], device=device)

def index_copy(xs, x):
    xs.index_copy_(0, torch.tensor(0).to(device), x)

# comment out to run in eager mode
index_copy = torch.compile(index_copy)

print("Before index copy")
print(f"x: {x}")
print(f"xs: {xs}")

index_copy(xs, x)

print()
print("After index copy")
print(f"xs: {xs}")