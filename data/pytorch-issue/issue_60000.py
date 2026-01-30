import mkl
mkl.set_num_threads(20)
mkl.set_dynamic(False)
mkl.get_max_threads()

import mkl
mkl.set_num_threads(20)
mkl.get_max_threads()

import torch
print(torch.get_num_threads())