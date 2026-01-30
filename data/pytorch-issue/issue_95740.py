import torch

inf = float('inf')
a = torch.tensor(complex(inf, inf), device=torch.device('cpu'))
b = torch.tensor(complex(inf, inf), device=torch.device('cuda'))
print(torch.exp(a))  # nan + nanj (wrong)
print(torch.exp(b))  # inf + nanj (correct)