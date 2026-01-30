import torch
import numpy as np

In [39]: len(torch.arange(184).chunk(18))
Out[39]: 17

arr1 = np.array([[1.,2,3], [4,5,6], [7,8,9]])
tsr1 = torch.tensor(arr1).cuda()

tsr_chuck = torch.chunk(tsr1, 
                        chunks = 4,
                        dim = 0)

arr_split = np.array_split(arr1,
                           indices_or_sections = 4,
                           axis = 0)

tsr_chuck

(tensor([[1., 2., 3.]], device='cuda:0', dtype=torch.float64),
 tensor([[4., 5., 6.]], device='cuda:0', dtype=torch.float64),
 tensor([[7., 8., 9.]], device='cuda:0', dtype=torch.float64))

arr_split

[array([[1., 2., 3.]]),
 array([[4., 5., 6.]]),
 array([[7., 8., 9.]]),
 array([], shape=(0, 3), dtype=float64)]