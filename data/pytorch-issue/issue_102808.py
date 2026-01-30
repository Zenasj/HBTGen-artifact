from torch import arange  
import torch   
lam = [1, 2, 3, 4, 5, 6]  
il = arange(len(lam))  
matrix= torch.sparse_csr_tensor(il, il, lam)  
print(matrix.to_dense())

tensor([[1, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 5, 6]])

from numpy import arange
from scipy.sparse import csr_matrix as sparse
lam = [1, 2, 3, 4, 5, 6]
il = arange(len(lam))
matrix = sparse((lam, (il, il)))
print(matrix.toarray())

from torch import arange  
import torch   
lam = [1, 2, 3, 4, 5, 6]  
il = list(range(len(lam)) )
matrix= torch.sparse_coo_tensor([il, il], lam).to_sparse(layout=torch.sparse_csr)  
print(matrix.to_dense())

tensor([[1, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0],
        [0, 0, 3, 0, 0, 0],
        [0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 5, 0],
        [0, 0, 0, 0, 0, 6]])