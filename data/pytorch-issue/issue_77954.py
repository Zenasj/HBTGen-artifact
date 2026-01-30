import torch

A = torch.randn(4, 4)
Ainv = torch.linalg.inv(A)
torch.dist(A @ Ainv, torch.eye(4))

A = torch.randn(2, 3, 4, 4)  # Batch of matrices
Ainv = torch.linalg.inv(A)
torch.dist(A @ Ainv, torch.eye(4))

A = torch.randn(4, 4, dtype=torch.complex128)  # Complex matrix
Ainv = torch.linalg.inv(A)
torch.dist(A @ Ainv, torch.eye(4))

def squareCheckInputs(self, f_name):
    assert self.dim() >= 2, f"{f_name}: The input tensor must have at least 2 dimensions."
    # TODO: I think the error message has the -2 and -1 swapped.  If you fix
    # it fix the C++ squareCheckInputs too
    assert self.size(-1) == self.size(-2), \
        f"{f_name}: A must be batches of square matrices, but they are {self.size(-1)} by {self.size(-2)} matrices"