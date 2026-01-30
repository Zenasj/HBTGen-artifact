import torch

A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
b = torch.tensor([[1, 2, 3]], dtype=torch.float32).t()
(LU_data, LU_pivots) = torch.lu(A)

x = torch.lu_solve(b, LU_data, LU_pivots)
print(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

if torch.cuda.is_available():
    A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device=device)
    b = torch.tensor([[1, 2, 3]], dtype=torch.float32, device=device).t()

    LU_data, LU_pivots = torch.lu(A)

    x = torch.lu_solve(b, LU_data, LU_pivots)

    x_cpu = x.cpu()
    print(x_cpu)