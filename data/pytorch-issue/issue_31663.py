import torch

n = 3000
m = 200

# torch.mv
nm = torch.randn((m, n), device=device).t()
_m = torch.randn((), device=device).expand(m)
_m_out = torch.full((m,), 0., device=device)

r0 = torch.mv(nm, _m)
r1 = torch.mv(nm, _m, out=_m_out)
# only NaNs differ
self.assertEqual(r0[torch.logical_or(r0 != r0, r1 != r1)].fill_(0),
                                 r1[torch.logical_or(r0 != r0, r1 != r1)].fill_(0))
self.assertEqual(torch.mv(nm, _m), torch.mv(nm, _m, out=_m_out))