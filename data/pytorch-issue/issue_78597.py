# module 'torch' has no attribute '_torch_docs'
import torch._torch_docs
torch._torch_docs.common_args

# works
import torch._torch_docs as t
t.common_args