import numpy as np
import torch

arr_symmetric = np.array([[1.,2,3], [2,5,6], [3,6,9]])
arr_symmetric, arr_symmetric.dtype

(array([[1., 2., 3.],
        [2., 5., 6.],
        [3., 6., 9.]]), dtype('float64'))

tsr_symmetric = torch.tensor(arr_symmetric)
tsr_symmetric

tensor([[1., 2., 3.],
        [2., 5., 6.],
        [3., 6., 9.]], dtype=torch.float64)

w, v = np.linalg.eigh(arr_symmetric)
w, v

(array([4.05517871e-16, 6.99264746e-01, 1.43007353e+01]),
 array([[-9.48683298e-01,  1.77819106e-01, -2.61496397e-01],
        [ 2.22044605e-16, -8.26924214e-01, -5.62313386e-01],
        [ 3.16227766e-01,  5.33457318e-01, -7.84489190e-01]]))

e, v = torch.symeig(tsr_symmetric, eigenvectors=True)
e, v

(tensor([-2.6047e-16,  6.9926e-01,  1.4301e+01], dtype=torch.float64),
 tensor([[ 9.4868e-01, -1.7782e-01,  2.6150e-01],
         [ 8.6389e-16,  8.2692e-01,  5.6231e-01],
         [-3.1623e-01, -5.3346e-01,  7.8449e-01]], dtype=torch.float64))