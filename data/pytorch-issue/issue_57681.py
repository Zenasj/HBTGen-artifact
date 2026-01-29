# torch.rand(2, 8, dtype=torch.complex64)  # Input shape: (2, 8) complex tensors for multiplication/division
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.avx2_mul = AVX2Mul()
        self.avx512_mul = AVX512Mul()
        self.avx2_div = AVX2Div()
        self.avx512_div = AVX512Div()
    
    def forward(self, inputs):
        x, y = inputs[0], inputs[1]
        # Multiplication comparison
        out_mul_avx2 = self.avx2_mul(x, y)
        out_mul_avx512 = self.avx512_mul(x, y)
        mul_close = torch.all(torch.isclose(out_mul_avx2, out_mul_avx512, atol=1e-6))
        
        # Division comparison
        out_div_avx2 = self.avx2_div(x, y)
        out_div_avx512 = self.avx512_div(x, y)
        div_close = torch.all(torch.isclose(out_div_avx2, out_div_avx512, atol=1e-6))
        
        return torch.logical_and(mul_close, div_close)

class AVX2Mul(nn.Module):
    def forward(self, x, y):
        return x * y

class AVX512Mul(nn.Module):
    def forward(self, x, y):
        real = x.real * y.real - x.imag * y.imag
        imag = x.real * y.imag + x.imag * y.real
        return torch.complex(real, imag)

class AVX2Div(nn.Module):
    def forward(self, x, y):
        return x / y

class AVX512Div(nn.Module):
    def forward(self, x, y):
        denom = y.real**2 + y.imag**2
        real = (x.real * y.real + x.imag * y.imag) / denom
        imag = (x.imag * y.real - x.real * y.imag) / denom
        return torch.complex(real, imag)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 8, dtype=torch.complex64)

