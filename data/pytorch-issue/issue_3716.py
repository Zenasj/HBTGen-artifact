#!/usr/bin/env python

import torch

for sz in [ 127, 128, 129, 130, 500 ]:
        print("Size = {}".format(sz))
        for i in range(10):
                a = torch.randn(sz, sz)
                q, r = torch.qr(a)
                a_qr = torch.mm(q, r)
                m = float(max(max(x) for x in a - a_qr))
                print("max diff = {0:.6f}{1}".format(m, " FAIL!" if m > 0.001 else ""))

import torch
a = torch.randn(1000,1000)
q,r = torch.qr(a)
m,tau = torch.geqrf(a)
a_qr = torch.orgqr(m,tau)
m_triu = torch.triu(m)
y = float(max(max(x) for x in q - a_qr))
z = float(max(max(x) for x in r - m_triu))

print("Q max diff = {0:.6f}{1}".format(y, " FAIL!" if y > 0.001 else ""))
print("R max diff = {0:.6f}{1}".format(z, " FAIL!" if z > 0.001 else ""))