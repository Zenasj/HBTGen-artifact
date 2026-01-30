import torch

py
#!/usr/bin/env python3

import sympy
from torch._inductor.codegen.triton import texpr
from torch._inductor.codegen.cpp import cexpr

s1 = sympy.Symbol('foo', integer=True)
expr = s1/(2.0*s1 - 1) - 1/(2.0*s1 - 1)

print(type(expr))
# sympy.core.add.Add

# Default pprinter:
print(str(expr))
# 'foo/(2.0*foo - 1) - 1/(2.0*foo - 1)'

# Triton
print(texpr(expr))
# '((-1)*) + (foo*)'

# CPP (uses ExprPrinter as texpr)
print(cexpr(expr))
#  '((-1)*) + (foo*)'