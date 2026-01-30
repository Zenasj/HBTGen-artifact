import torch
import torch.nn as nn

code_string = "template <typename T> T inverted_where(bool cond, T a, T b){ return !cond ? a : b; }"
jitted_fn = torch.cuda.jiterator._create_jit_fn(code_string) 
my_lib = torch.library.Library("aten", "IMPL")
my_lib.impl('aten::where.self', jitted_fn, "CUDA")

# torch.where is now overridden

code_string = "template <typename T> T fast_gelu(T a){ return a > 0 ? a : 0;}"
jitted_fn = torch.cuda.jiterator._create_jit_fn(code_string) 
my_lib = torch.library.Library("aten", "IMPL")
my_lib.impl('aten::gelu', jitted_fn, "CUDA")

# torch.nn.GELU and torch.nn.function.gelu are now overridden

code_string = "template <typename T> T clipped_exp(T a){ return a > T(10.0) ? T(22026.4657948) : exp(a); }"
jitted_fn = torch.cuda.jiterator._create_jit_fn(code_string) 
my_lib = torch.library.Library("aten", "IMPL")
my_lib.impl('aten::exp', jitted_fn, "CUDA")

# torch.exp(x) and x.exp() are now overridden