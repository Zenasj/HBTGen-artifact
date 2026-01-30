import torch
inf = float("inf")

x = torch.tensor([[-inf, 1]])

def logcumsumexp_helper(x, dim):
    y_cpu = torch.logcumsumexp(x, dim=dim)
    y_cuda = torch.logcumsumexp(x.cuda(), dim=dim)

    print("CPU:")
    print(y_cpu)
    print("CUDA:")
    print(y_cuda)
    print("DIFF:")
    print(y_cpu - y_cuda.cpu())

print("CUB Path")
logcumsumexp_helper(x, dim=1)
print("\n\n")

print("Innermost path")
x = torch.tensor([
    [2., -inf, -inf, 1, 3],
    [2., 1,    -inf, 1, 3]
])
logcumsumexp_helper(x, dim=-1)
print("\n\n")

print("Outermost path")

logcumsumexp_helper(x, dim=0)
print("\n\n")