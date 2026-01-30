import torch

A = torch.load('A.ct')
input_ = torch.load('input.ct')

print(f"A.shape: {A.shape}")
print(f"input.shape: {input_.shape}")

assert (A[0] == A[1]).all()
assert (input_[0] == input_[1]).all()

solution = torch.triangular_solve(input_, A, upper=False)[0]

print(f"(solution[0] == solution[1]).all() {(solution[0] == solution[1]).all()}")

solution_0 = torch.triangular_solve(input_[0], A[0], upper=False)[0]
solution_1 = torch.triangular_solve(input_[1], A[1], upper=False)[0]

print(f"(solution_0 == solution_1).all() {(solution_0 == solution_1).all()}")