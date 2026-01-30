import torch

@skipCUDAIfRocm
@dtypesIfCUDA(torch.half, torch.float, torch.double)
@dtypes(torch.float, torch.double)
def test_blas_alpha_beta_empty(self, device, dtype):
    # ensure beta is respected
    value = 11
    input = torch.full((2,), value, dtype=dtype, device=device)
    mat = torch.ones((2, 0), dtype=dtype, device=device)
    vec = torch.ones((0,), dtype=dtype, device=device)
    out = torch.empty((2,), dtype=dtype, device=device)
    alpha = 6
    beta = 3
    self.assertEqual(torch.full((2,), beta * value, dtype=dtype, device=device),
                        torch.addmv(input=input, mat=mat, vec=vec, alpha=alpha, beta=beta))
    self.assertEqual(torch.full((2,), beta * value, dtype=dtype, device=device),
                        torch.addmv(input=input, mat=mat, vec=vec, alpha=alpha, beta=beta, out=out))

    # torch.addmm
    input = torch.full((2, 3), value, dtype=dtype, device=device)
    mat2 = torch.ones((0, 3), dtype=dtype, device=device)
    out = torch.empty((2, 3), dtype=dtype, device=device)
    self.assertEqual(torch.full((2, 3), beta * value, dtype=dtype, device=device),
                        torch.addmm(input=input, mat1=mat, mat2=mat2, alpha=alpha, beta=beta))
    self.assertEqual(torch.full((2, 3), beta * value, dtype=dtype, device=device),
                        torch.addmm(input=input, mat1=mat, mat2=mat2, alpha=alpha, beta=beta, out=out))