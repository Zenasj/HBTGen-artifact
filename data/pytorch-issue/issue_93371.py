import torch.nn as nn

import torch

def fn(input):
    v1 = torch.nn.functional.softmax(input, 1) # works fine w/o this line
    v2 = v1.transpose(0, 3) # works fine w/o this line
    return v2.div_(2.0) # works fine with non-inplace operator "div"

x = torch.rand([4, 6, 4, 1])

ret_eager = fn(x)
print('==== Eager mode OK! ====')

compiled = torch.compile(fn)
print('==== torchcomp compilation OK! ====')

ret_compiled = compiled(x)
print('==== torchcomp mode OK! ====')