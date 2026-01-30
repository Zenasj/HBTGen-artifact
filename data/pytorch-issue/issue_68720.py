import torch

class _PositiveDefinite(Constraint):
    """
    Constrain to positive-definite matrices.
    """
    event_dim = 2

    def check(self, value):
        # Assumes that the matrix or batch of matrices in value are symmetric
        # info == 0 means no error, that is, it's SPD
        return torch.linalg.cholesky_ex(value).info.eq(0).unsqueeze(0)

class _PositiveDefinite(Constraint):
    """
    Constrain to positive-definite matrices.
    """
    event_dim = 2

    def check(self, value):
        return (torch.linalg.eig(value)[0].real > 0).all(dim=-1)