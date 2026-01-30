import torch
torch.manual_seed(126)
n = 5
kwargs = {
    "dtype": torch.float64,
}
# random matrix to be orthogonalized for the eigenvectors
mat = torch.randn((n, n), **kwargs).requires_grad_()

# matrix for the loss function
P2 = torch.randn((n, n), **kwargs).requires_grad_()

# the degenerate eigenvalues
a = torch.tensor([0.5, 1.0, 2.0], **kwargs).requires_grad_()

def get_loss(a, mat, P2):
    # get the orthogonal vector for the eigenvectors
    P, _ = torch.qr(mat)

    # line up the eigenvalues
    b = torch.cat((a[:2], a[1:2], a[2:], a[2:]))

    # construct the matrix
    diag = torch.diag_embed(b)
    A = torch.matmul(torch.matmul(P.T, diag), P)

    eivals, eivecs = torch.symeig(A, eigenvectors=True)
    U = eivecs[:, 1:3]  # the degenerate eigenvectors

    loss = torch.einsum("rc,rc->", torch.matmul(P2, U), U)
    return loss

torch.autograd.gradcheck(get_loss, (a, mat, P2))