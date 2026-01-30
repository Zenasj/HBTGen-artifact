import torch

py
def test_rpow_scalar_float_shape():
    assert (4 ** torch.tensor(0.5)).shape == ()