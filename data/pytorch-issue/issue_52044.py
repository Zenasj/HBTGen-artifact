import scipy.optimize
import torch

def linear_assignment(C):
    rowinds, colinds = zip(*list(map(scipy.optimize.linear_sum_assignment, C.cpu())))
    print(colinds[0].shape, colinds[0].dtype)
    # (8,) int64
    return torch.stack(list(map(torch.as_tensor, colinds)), out = torch.empty(C.shape[:2], dtype = torch.int64, device = C.device))
    # return torch.stack(list(map(torch.as_tensor, colinds))).to(C.device) # works


if __name__ == '__main__':
    C = torch.rand(31, 8, 8, device = 'cuda')
    P = linear_assignment(C)
    print(P.shape, P.dtype)
    # torch.Size([31, 8]) torch.int64
    print(P)
#    Traceback (most recent call last):
#  File "bug.py", line 13, in <module>
#    print(P)
#  File "/vadim/prefix/miniconda/lib/python3.8/site-packages/torch/tensor.py", line 179, in __repr__
#    return torch._tensor_str._str(self)
#  File "/vadim/prefix/miniconda/lib/python3.8/site-packages/torch/_tensor_str.py", line 372, in _str
#    return _str_intern(self)
#  File "/vadim/prefix/miniconda/lib/python3.8/site-packages/torch/_tensor_str.py", line 352, in _str_intern
#    tensor_str = _tensor_str(self, indent)
#  File "/vadim/prefix/miniconda/lib/python3.8/site-packages/torch/_tensor_str.py", line 241, in _tensor_str
#    formatter = _Formatter(get_summarized_data(self) if summarize else self)
#  File "/vadim/prefix/miniconda/lib/python3.8/site-packages/torch/_tensor_str.py", line 85, in __init__
#    value_str = '{}'.format(value)
#  File "/vadim/prefix/miniconda/lib/python3.8/site-packages/torch/tensor.py", line 534, in __format__
#    return self.item().__format__(format_spec)
#RuntimeError: CUDA error: an illegal memory access was encountered