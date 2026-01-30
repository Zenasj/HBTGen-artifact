import torch

lr, lambd, alpha = 1e-2, 1e-4, 0.75 # default values of ASGD
n1 = lr / (1 + lambd * lr * 1**alpha)
n2 = lr / (1 + lambd * lr * 2**alpha)

print(n1, n2)  # 0.009999990000010001 0.00999998318209998
print(n1 == n2)  # False
print(torch.as_tensor(n1),
      torch.as_tensor(n2))  # tensor(0.0100) tensor(0.0100)
print(
    torch.allclose(torch.as_tensor(n1),
                   torch.as_tensor(n2)))  # True
print(torch.as_tensor(n1, dtype=torch.float64),
      torch.as_tensor(n2, dtype=torch.float64))  # tensor(0.0100) tensor(0.0100)
print(
    torch.allclose(torch.as_tensor(n1, dtype=torch.float64),
                   torch.as_tensor(n2, dtype=torch.float64)))  # True