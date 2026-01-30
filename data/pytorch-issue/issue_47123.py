import torch

from torch.distributions import Normal, Independent 

test_param = torch.tensor([[4.2, -8.1], [-12.4, 1.7]])
incorrect_normal = Normal(test_param, test_param)
wrapped_normal = Independent(incorrect_normal, 1)

# > incorrect_normal.stddev
# > tensor([[  4.2000,  -8.1000],
#         [-12.4000,   1.7000]])

# > wrapped_normal.stddev
# > tensor([[ 4.2000,  8.1000],
#         [12.4000,  1.7000]])

py
torch.distributions.Distribution.set_default_validate_args(True)