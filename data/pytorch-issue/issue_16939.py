3
import torch
from timeit import timeit

# Some defaults
number = 48000
n = 256
torch.set_grad_enabled(False)
print('PyTorch version: ', torch.__version__)
print('MKL: ', torch.backends.mkl.is_available())
print('Iterations: ', number)
print('Threads: ', torch.get_num_threads())
print('Size: ', n)

a = torch.rand(256)

# Test for performance
kwargs = {'globals': globals(), 'number': number}
print('torch.multinomial(a, 1):', timeit("torch.multinomial(a, 1)", **kwargs))