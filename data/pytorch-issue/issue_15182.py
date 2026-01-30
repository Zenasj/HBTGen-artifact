import torch  # 0.3.1

@profile
def test_btri(A, b):
    A_LU = A.btrifact()
    P, A_L, A_U = torch.btriunpack(*A_LU)
    torch.btrisolve(b, *A_LU)

if __name__ == '__main__':
    A = torch.randn(20, 30, 30).cpu()
    b = torch.randn(20, 30).cpu()

    test_btri(A, b)

import torch  # 1.0.0

@profile
def test_btri(A):
    A_LU = A.btrifact()
    P, A_L, A_U = torch.btriunpack(*A_LU)
    torch.btrisolve(b, *A_LU)

if __name__ == '__main__':
    with torch.no_grad():
        A = torch.randn(20, 30, 30).cpu()
        b = torch.randn(20, 30).cpu()

        test_btri(A)