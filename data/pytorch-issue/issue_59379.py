# torch.rand(B, 3, 1, 1, dtype=torch.int32)
import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, input_params):
        # Extract parameters from input tensor (shape: B,3,1,1)
        rows = input_params[0, 0, 0].item()
        cols = input_params[0, 1, 0].item()
        nnz = input_params[0, 2, 0].item()
        size = (rows, cols)
        
        # Generate CSR tensors using both methods
        t = TestCase()
        # Current method (problematic genSparseCSRTensor)
        a = t.genSparseCSRTensor(size, nnz, device='cpu', dtype=torch.float32, index_dtype=torch.int32)
        crow_a = a.crow_indices()
        
        # Alternative method (genSparseTensor -> to_sparse_csr)
        b = t.genSparseTensor(size, 2, nnz, False, 'cpu', torch.float32)[0].to_sparse_csr()
        crow_b = b.crow_indices()
        
        # Compare crow indices and return difference as tensor
        result = torch.any(crow_a != crow_b).to(torch.int32).unsqueeze(0)
        return result

def my_model_function():
    return MyModel()

def GetInput():
    import random
    # Randomly choose between test cases from the issue's examples
    sizes = [(10, 10), (15, 20)]
    nnzs = [10, 40]
    size = random.choice(sizes)
    nnz = random.choice(nnzs)
    rows, cols = size
    # Create input tensor matching the expected shape (B=1, C=3, H=1, W=1)
    input_tensor = torch.tensor([rows, cols, nnz], dtype=torch.int32).view(1, 3, 1, 1)
    return input_tensor

