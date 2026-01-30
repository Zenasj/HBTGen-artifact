import torch

feats = torch.ones((2810, 2053), device="cuda")

adj = torch.zeros((2810, 2810), device="cuda")
adj[torch.randint(0, 2810, size=(15962, )), torch.randint(0, 2810, size=(15962, ))] = 1

w = torch.ones((2053, 7), device="cuda")

a = (adj @ feats) @ w
print(f"a = {a}")

b = adj @ (feats @ w)
print(f"b = {b}")

print(torch.allclose(a, b))