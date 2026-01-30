import torch
import numpy as np

A = torch.Tensor([[1,2,3], [4,5,6]])
B = A.numpy().copy()

print(A) # [[1,2,3],[4,5,6]]
print(B) # [[1,2,3],[4,5,6]]

# Subtract first column from all columns
A -= A[:,0][:,np.newaxis] # or unsqueeze(1)
B -= B[:,0][:,np.newaxis]

print(A) # [[0,2,3],[0,5,6]] (unexpected result / bug)
print(B) # [[0,1,2],[0,1,2]] (this is expected result in my opinion)