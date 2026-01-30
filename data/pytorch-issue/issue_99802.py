import torch
x = torch.tensor([0.5, 0.5], dtype=torch.float, device='cpu')
print(set(torch.multinomial(x, 1, True).item() for i in range(100)))
x = torch.tensor([0.5, 0.5], dtype=torch.float, device='mps')
print(set(torch.multinomial(x, 1, True).item() for i in range(100)))

# Output:
# {0, 1}
# {0, 1}