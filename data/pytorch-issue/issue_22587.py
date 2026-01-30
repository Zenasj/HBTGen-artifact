#!/usr/bin/env python3

import torch

# All functions based on 1998__Stewart__Matrix_Algorithms-Basic_Decompositions

@torch.jit.script
def rotgen(a_in, b_in):
    r = abs(a_in) + abs(b_in)
    if bool(r == 0.0):
        a_out = a_in.clone()
        b_out = b_in.clone()
        c_out = torch.ones_like(a_out, dtype=torch.double)
        s_out = torch.zeros_like(a_out, dtype=torch.double)
    else:
        a_out = r * ((a_in / r)**2 + (b_in / r)**2)**0.5
        b_out = torch.zeros_like(a_out, dtype=torch.double)
        c_out = a_in / a_out
        s_out = b_in / a_out
    return a_out, b_out, c_out, s_out

@torch.jit.script
def rotapp(x_in, y_in,
           c, s):
    x_out = c * x_in + s * y_in
    y_out = c * y_in - s * x_in
    return x_out, y_out

@torch.jit.script
def cholupdate(R_in, v_in):
    R_out = R_in.clone()
    v_out = v_in.clone()
    
    p = len(v_in)
    for k in range(p):
        R_out[k, k], v_out[k], c, s = rotgen(R_out[k, k], v_out[k])
        R_out[k, k+1:p], v_out[k+1:p] = rotapp(R_out[k, k+1:p], v_out[k+1:p],
                                               c, s)
        #print(v_out)
    return R_out

@torch.jit.script
def cholxdate(R, v, w):
    R_out = R.transpose(1, 0)
    v_use = v * float(abs(w)**0.25)

    R_out = cholupdate(R_out, v_use)
    R_out = R_out.transpose(1, 0)
    return R_out
            
if __name__ == '__main__':
    import numpy as np
    
    dim = 2

    # manually set the variables
    w = np.array([1.0], dtype=np.float64)
    v = np.full(dim, np.sqrt(1/3), dtype=np.float64)
    R = np.zeros((dim, dim), dtype=np.float64)
    R[np.tril_indices(dim)] = np.arange(1, np.cumsum(range(dim+1))[-1]+1, dtype=np.float64)
    M = np.dot(R, R.T)
    
    # make the variables torch tensors
    M = torch.from_numpy(M)
    R = torch.from_numpy(R)
    v = torch.from_numpy(v)
    w = torch.from_numpy(w)

    # cholesky update
    M_up0 = M + torch.sqrt(w) * torch.ger(v, v)
    R_up = cholxdate(R, v, w)
    M_up = torch.mm(R_up, R_up.transpose(1, 0))
    R_up0 = torch.cholesky(M_up0)
    print('cholesky update:')
    print('M_up')
    print(M_up)
    print('M_up0')
    print(M_up0)
    print('R_up')
    print(R_up)
    print('R_up0')
    print(R_up0)