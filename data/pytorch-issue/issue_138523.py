import sympy
import torch

from torch.utils._sympy.functions import FloorDiv
s0 = sympy.Symbol('s0')

expr = FloorDiv(s0, 2)
print(expr)
cpp_code = sympy.cxxcode(expr)
print(cpp_code)

cpp_code2 = sympy.ccode(expr)
print(cpp_code2)