import torch.nn as nn

import torch

class DeterministicModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(2))

    def forward(self, x):
        block_diag = torch.block_diag(x, x)
        LD = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        pivots = torch.tensor([0, 0], dtype=torch.int32)
        B = torch.tensor([[1.0], [1.0]])
        solution = torch.linalg.ldl_solve(LD, pivots, B)
        unique_solution = torch.unique_consecutive(solution.flatten())
        return block_diag, solution, unique_solution

model = DeterministicModel()
inputs = torch.eye(2)
res = model(inputs)