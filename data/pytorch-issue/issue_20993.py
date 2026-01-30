import torch

x = _l2_norm(x, 1)
cosine = torch.matmul(x, x.transpose(0, 1))
sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

def _l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output