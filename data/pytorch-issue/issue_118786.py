from hypothesis import given
import torch
import torch.testing._internal.hypothesis_utils as hu

@given(X=hu.tensor(shapes=hu.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=1)))
def test_avg_pool2d_nhwc(X):
    pass

torch.compile(test_avg_pool2d_nhwc)()