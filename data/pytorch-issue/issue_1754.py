x = torch.rand(10, 2).cuda()
complex_cube(x)

# inspired by https://gist.github.com/szagoruyko/89f83b6f5f4833d3c8adf81ee49f22a8

from functools import lru_cache
import torch

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

CUDA_NUM_THREADS = 1024

def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS

class Holder(cuda.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        assert t.is_cuda
        self.t = t
        self.gpudata = t.data_ptr()
    def get_pointer(self):
        return self.t.data_ptr()

def complex_square(x):
    assert x.is_cuda
    assert x.size(-1) == 2
    batch_size = x.size()[:-1]

    x = x.view(-1, 2) # [batch, complex] (nbatch, 2)
    nbatch = x.size(0)

    cuda_kernel = _setup_complex_square_cuda_kernel(nbatch)

    output = torch.cuda.FloatTensor(nbatch, 2)
    cuda_kernel(Holder(x), Holder(output),
                block=(CUDA_NUM_THREADS, 1, 1),
                grid=(GET_BLOCKS(nbatch), 1, 1))
    # [batch, complex] (nbatch, 2)

    output = output.view(*batch_size, 2)
    return output

@lru_cache(maxsize=32)
def _setup_complex_square_cuda_kernel(nbatch):
    kernel = '''
extern "C"
__global__ void main_(const float* in, float* out) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < {nbatch}; index += blockDim.x * gridDim.x) {
        float in_re = in[index * 2 + 0];
        float in_im = in[index * 2 + 1];
        out[index * 2 + 0] = in_re * in_re - in_im * in_im;
        out[index * 2 + 1] = 2.0 * in_re * in_im;
    }
}
'''.replace("{nbatch}", str(nbatch))
    
    return SourceModule(kernel).get_function("main_")

def complex_cube(x):
    assert x.is_cuda
    assert x.size(-1) == 2
    batch_size = x.size()[:-1]

    x = x.view(-1, 2) # [batch, complex] (nbatch, 2)
    nbatch = x.size(0)

    cuda_kernel = _setup_complex_cube_cuda_kernel(nbatch)

    output = torch.cuda.FloatTensor(nbatch, 2)
    cuda_kernel(Holder(x), Holder(output),
                block=(CUDA_NUM_THREADS, 1, 1),
                grid=(GET_BLOCKS(nbatch), 1, 1))
    # [batch, complex] (nbatch, 2)

    output = output.view(*batch_size, 2)
    return output

@lru_cache(maxsize=32)
def _setup_complex_cube_cuda_kernel(nbatch):
    kernel = '''
extern "C"
__global__ void main_(const float* in, float* out) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < {nbatch}; index += blockDim.x * gridDim.x) {
        float in_re = in[index * 2 + 0];
        float in_im = in[index * 2 + 1];
        out[index * 2 + 0] = in_re * (in_re * in_re - 3.0 * in_im * in_im);
        out[index * 2 + 1] = in_im * (3.0 * in_re * in_re - in_im * in_im);
    }
}
'''.replace("{nbatch}", str(nbatch))

    return SourceModule(kernel).get_function("main_")

class FooBar(torch.autograd.Function):
    def __init__(self):
        super(FooBar, self).__init__()

    def forward(self, x):
        return complex_square(x)

    def backward(self, grad_output):
        return complex_cube(grad_output)

from torch.autograd import Variable

x = torch.rand(100, 100, 2).cuda()
x = Variable(x, requires_grad=True)

op = FooBar()
y = op(x)

z = y.sum()
z.backward()

from torch.autograd import Variable

x = torch.rand(100, 100, 2).cuda()
x = Variable(x, requires_grad=True)

op = FooBar()
y = op(x)

complex_cube(y.data) # this line is added

z = y.sum()
z.backward()