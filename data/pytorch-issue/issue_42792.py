import torch

def pos_qr(A):
    """ Returns the QR decomposition with R.diag() > 0 """
    Q, R = torch.qr(A)
    d = torch.diag_embed(torch.diag(R).sign())
    # make fast
    return Q @ d, d @ R


def same_function_different_grad(A, G, conj):
    """ Compute one of: 
       - torch.qr(A).Q
       - U.t() @ torch.qr(U @ A).Q
    for an orthogonal U. 
    In both cases the result should be the same
    """
    if conj:
        U = torch.qr(torch.rand(8, 8)).Q
        A = U @ A
    Q, _ = pos_qr(A)
    if conj:
        Q = U.t() @ Q
    out = torch.autograd.grad(Q, A, G)[0]
    return Q, out

# A is not full-rank by construction
A = torch.cat([torch.rand(8, 3, requires_grad=True), torch.zeros(8, 5, requires_grad=True)], dim=1)
# Some fixed gradient
G = torch.rand(8, 8)
Q1, B1 = same_function_different_grad(A, G, False)
Q2, B2 = same_function_different_grad(A, G, True)
print(torch.norm(Q1 - Q2))
print(torch.norm(B1 - B2))