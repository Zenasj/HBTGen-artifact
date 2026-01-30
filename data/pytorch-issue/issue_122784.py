import torch
import torch.nn as nn

def bmm1(m1, m2):
    return torch.bmm(m1,m2.unsqueeze(2)).squeeze_(2)

def bmm2(m1,m2):
    return torch.sum(m1 * m2.unsqueeze(1), dim=2)


class MEVP(torch.nn.Module):
    def __init__(self):
        super(MEVP, self).__init__()

    # iMJwPSI [1,3,4], rhs [1,4]
    def forward(self, iMJwPSI, rhs):
        res1 = bmm1(iMJwPSI, rhs)
        res2 = bmm2(iMJwPSI, rhs)

        return res1, res2