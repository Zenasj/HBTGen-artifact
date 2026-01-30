import torch

a = torch.tensor([-7742.+0.j,-15601.+0.j,-30536.+0.j,-26006.+0.j,-9821.+0.j,-19432.+0.j,-20112.+0.j,-9278.+0.j, -25131.+0.j, -14546.+0.j,-8500.+0.j,-13001.+0.j,-17000.+0.j,-12000.+0.j,-6000.+0.j,-20500.+0.j], dtype=torch.complex64)

out_cpu = torch.sigmoid(a)
print(out_cpu)
# tensor([nan+nanj, nan+nanj, nan+nanj, nan+nanj, nan+nanj, nan+nanj, nan+nanj, nan+nanj, nan+nanj, nan+nanj, nan+nanj, nan+nanj, nan+nanj, nan+nanj,
#         nan+nanj, nan+nanj])

out_gpu = torch.sigmoid(a.cuda())
print(out_gpu)
# tensor([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
#         0.+0.j, 0.+0.j], device='cuda:0')

import numpy as np

np_array = np.array([-7742.+0.j,-15601.+0.j,-30536.+0.j,-26006.+0.j,-9821.+0.j,-19432.+0.j,-20112.+0.j,-9278.+0.j, -25131.+0.j, -14546.+0.j,-8500.+0.j,-13001.+0.j,-17000.+0.j,-12000.+0.j,-6000.+0.j,-20500.+0.j])

print(1/(1+np.exp(-np_array)))
# [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j
#  0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
# <ipython-input-24-a5a1955c34c4>:5: RuntimeWarning: overflow encountered in exp
#   print(1/(1+np.exp(-np_array)))