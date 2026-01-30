import torch.nn as nn

import torch
import numpy as np

m = torch.nn.LeakyReLU(0.1)

scale=0.061773329973220825
zero = 126

a = np.arange(zero-63,zero).astype(np.float32)

a = (a-zero)*scale
b = np.append(a,a[-1])

print(f'a shape: {a.shape}')
print(f'b shape: {b.shape}')
print()

ta = torch.quantize_per_tensor(torch.tensor(a), scale, zero, torch.quint8)
tb = torch.quantize_per_tensor(torch.tensor(b), scale, zero, torch.quint8)

ta = m(ta)
tb = m(tb)

print('In   OutA  OutB')
for i in range(0,63):
    if ta[i].int_repr() != tb[i].int_repr():
        print(f'{zero-63+i} : {ta[i].int_repr()} , {tb[i].int_repr()} <<')
    else:
        print(f'{zero-63+i} : {ta[i].int_repr()} , {tb[i].int_repr()}')