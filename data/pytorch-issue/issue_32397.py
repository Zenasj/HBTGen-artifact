import torch.nn as nn
import random

import torch, scipy, random
from scipy.stats import kstest
from torch.nn import init
def _is_trunc_normal(tensor, mean, std, a, b):
    p_value = scipy.stats.kstest(tensor.flatten().tolist(), 'truncnorm', args=(a, b))[1]
    return p_value


if __name__ == '__main__':
    input_tensor = torch.empty((10, 10, 20))
    for _ in range(1000):
        a = random.uniform(3, 3)
        b = random.uniform(a, a + 1)
        init.trunc_normal_(input_tensor.flatten(), mean=0., std=1., a=a, b=b)

        p_value = _is_trunc_normal(input_tensor, 0., 1., a, b)
        if p_value <= 0.0001:
            print("Failed for interval [{0:.3}, {1:.3}], length {2:.3}".format(a, b, b-a))

# Failed for interval [4.98, 5.18], length 0.202
# Failed for interval [3.87, 3.87], length 0.00138
# Failed for interval [2.89, 2.89], length 0.000154

t = torch.zeros((1, 10))
init.trunc_normal_(t, 3, 0.1, -2, 2) # mean is 3, std is .1, truncated to [-2, 2]
# __main__:1: UserWarning: mean is more than 2 std from [a, b] in nn.init.trunc_normal_.
# The distribution of values may be incorrect.