# torch.rand(2, 2, dtype=torch.float64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        original = x
        # CSR conversion chain: CSR → COO → CSC → COO → CSR
        csr = x.to_sparse_csr()
        converted_csr = csr.to_sparse_coo().to_sparse_csc().to_sparse_coo().to_sparse_csr()
        result_csr = converted_csr.to_dense()
        csr_ok = torch.allclose(original, result_csr, atol=1e-7)

        # CSC conversion chain: CSC → COO → CSR → COO → CSC
        csc = x.to_sparse_csc()
        converted_csc = csc.to_sparse_coo().to_sparse_csr().to_sparse_coo().to_sparse_csc()
        result_csc = converted_csc.to_dense()
        csc_ok = torch.allclose(original, result_csc, atol=1e-7)

        # Return True only if both conversions are correct
        return torch.tensor([csr_ok and csc_ok], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2, dtype=torch.float64)

