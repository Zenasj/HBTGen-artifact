import torch

class FactoredMatrix:
  def __init__(self, mat):
    self.mat = mat
  def __matmul__(self, other):
    return 0
  def __rmatmul__(self, other):
    return 1
x = torch.ones((2,2))
print(FactoredMatrix(x) @ x)  # works
print(x @ FactoredMatrix(x))  # fails