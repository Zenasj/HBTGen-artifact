import torch


@torch.compile(dynamic=True)
def func(A: torch.Tensor, threshold=0.0):
    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        assert A.dim() == 2, f"Input tensor should be 2d or 3d but got {A.dim()}d"
        rows = A.shape[0]
    A = A.reshape(rows, cols)

    if threshold == 0.0:
        outlier_indices = None
        outlier_coord = None
        outlier_rows = None
        outlier_cols = None
        outlier_values = None
    else:
        outlier_indices = torch.abs(A) >= threshold
        outlier_coord = outlier_indices.nonzero()
        outlier_rows = outlier_coord[:, 0]
        outlier_cols = outlier_coord[:, 1]
        outlier_values = A[outlier_indices]

    return outlier_indices, outlier_coord, outlier_rows, outlier_cols, outlier_values


print('1')
A = torch.randn(2048, 2048) * 3
func(A)
print('2')
A = torch.randn(8192, 2048) * 3
func(A)
print('3')
A = torch.randn(2048, 8192) * 3
func(A)

print('4')
A = torch.randn(2048, 2048) * 3
func(A, threshold=3)
print('5')
A = torch.randn(8192, 2048) * 3
func(A, threshold=3)
print('6')
A = torch.randn(2048, 8192) * 3
func(A, threshold=3)

for i in range(10):
    print('run', i)
    bs = pow(2, i)
    A = torch.randn(bs, 2, 2048) * 3
    func(A, threshold=3)
print('ok')