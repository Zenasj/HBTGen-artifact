import torch

# Two tensors of different shapes
A = torch.zeros((1, 1))
B = torch.zeros((1,))

torch.testing.assert_allclose(A, B) # passes

import numpy as np

A = np.zeros((1, 1))
B = np.zeros((1,))

np.testing.assert_allclose(A, B) # fails

if expected.shape != actual.shape:
        expected = expected.expand_as(actual)

A = torch.zeros((1, 1))
B = torch.zeros((1,))

torch.testing.assert_allclose(A, B) # passes
torch.testing.assert_allclose(B, A) # fails