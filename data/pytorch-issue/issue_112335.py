import numpy as np
import torch

def test():
    return np.asarray([np.array([1, 2]), np.array([2, 3])])

def main():
    test_compiled = torch.compile(test)
    print(test())
    print(test_compiled())

if __name__ == '__main__':
    main()

from torch._dynamo import config
config.trace_numpy = False