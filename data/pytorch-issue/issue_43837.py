import random

import torch
import torch.distributions as td
from torch.autograd import profiler, grad

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


class StubMultivariateNormal(td.MultivariateNormal):
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        super().__init__(loc, covariance_matrix, precision_matrix, scale_tril, validate_args)
        # retry procedure with broadcasting
        loc_ = loc.unsqueeze(-1)  # temporarily add dim on right
        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.scale_tril, _ = torch.broadcast_tensors(scale_tril, loc_)
            self.loc, _ = torch.broadcast_tensors(loc, scale_tril[..., 0])
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.covariance_matrix, _ = torch.broadcast_tensors(covariance_matrix, loc_)
            self.loc, _ = torch.broadcast_tensors(loc, covariance_matrix[..., 0])
        else:
            if precision_matrix.dim() < 2:
                raise ValueError("precision_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.precision_matrix, _ = torch.broadcast_tensors(precision_matrix, loc_)
            self.loc, _ = torch.broadcast_tensors(loc, precision_matrix[..., 0])


if __name__ == '__main__':
    torch.random.manual_seed(1234)
    loc = torch.randn(1000, 800)
    target = torch.randn(1000, 800)
    cov = target.transpose(-1, -2) @ target / 1000
    loc.requires_grad_()
    dist1 = td.MultivariateNormal(loc=loc, covariance_matrix=cov)
    dist2 = StubMultivariateNormal(loc=loc, covariance_matrix=cov)
    
    # run with modified distribution first to ensure that warmup does not swindle the benchmark
    with profiler.profile() as prof:
        with profiler.record_function('dist2_grad'):
            grad(dist2.log_prob(target).sum(), loc)
    print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=5))
    
    # run with torch's MultivariateNormal
    with profiler.profile() as prof:
        with profiler.record_function('dist1_grad'):
            grad(dist1.log_prob(target).sum(), loc)
    print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=5))

py
shape = torch.broadcast_shape(loc.shape, scale_tril.shape[:-1])
loc = loc.expand(shape)
scale_tril = scale_tril.expand(shape + (-1,))

py
def naive_implementation(*shapes):
    result = [1] * max(len(shape) for shape in shapes)
    for shape in shapes:
        for i in range(-1, -1 - len(shape), -1):
            size = shape[i]
            if size != 1:
                result[i] = size
    return torch.Size(result)

# (good_inputs, expected_output)
GOOD_EXAMPLES = [
    ([(1, 2), (3, 1)], (3, 2)),
    ([(3, 2), (3, 1)], (3, 2)),
    ([(3, 1), (3, 2)], (3, 2)),
    ([(2, 1, 1), (1, 3, 1), (1, 1, 4)], (2, 3, 4)),
]

# bad_inputs
BAD_EXAMPLES = [
  [(2,), (3,)],
]

class TestBroadcastShape(TestCase):
    def test_ok(self):
        for inputs, output in GOOD_EXAMPLES:
            inputs = map(torch.Size, inputs)
            output = torch.Size(output)
            assert torch.broadcast_shape(*inputs) == naive_implementation(*inputs)

    def test_error(self):
        for inputs in BAD_EXAMPLES:
            inputs = map(torch.Size, inputs)
            with self.assertRaises(ValueError):
                torch.broadcast_shape(*inputs)

py
def broadcast_shapes(*shapes):
    scalar = torch.zeros((), device="cpu")  # or even torch.empty()
    tensors = [scalar.expand(shape) for shape in shapes]
    tensors = torch.broadcast_all(*tensors)
    return tensors[0].shape