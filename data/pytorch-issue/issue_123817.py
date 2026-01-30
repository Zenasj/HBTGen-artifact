import torch
import numpy as np

def test_leaky_relu():
    shape = (16, 5, 5)
    dtype = torch.bfloat16
    slope = 1.0 / np.sqrt(256)
    m = torch.ops.aten.leaky_relu
    torch._dynamo.reset()
    m = torch.compile(m)
    self = torch.randn(shape, dtype=dtype) * 10.0
    result = m(self, slope)
    print(result)

if __name__ == '__main__':
    test_leaky_relu()