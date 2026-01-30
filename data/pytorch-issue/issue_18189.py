import torch
import torch.nn as nn

def test_inplace():
    linear = nn.Linear(20,6)

    x_fwd = torch.randn(7,1,20)
    x_bwd = x_fwd.clone().flip(1)

    y_fwd = linear(x_fwd)
    y_bwd = linear(x_bwd)
    print((y_fwd-y_bwd).abs().sum().item())
    z_gate_fwd,f_gate_fwd,o_gate_fwd = y_fwd.chunk(3, dim=2)
    z_gate_bwd,f_gate_bwd,o_gate_bwd = y_bwd.chunk(3, dim=2)
    print((z_gate_fwd-z_gate_bwd).abs().sum().item())
    print((f_gate_fwd-f_gate_bwd).abs().sum().item())
    print((o_gate_fwd-o_gate_bwd).abs().sum().item())
    z_gate_fwd.tanh_()
    z_gate_bwd.tanh_()
    f_gate_fwd.sigmoid_()
    f_gate_bwd.sigmoid_()
    z_gate_fwd,f_gate_fwd = z_gate_fwd.contiguous(),f_gate_fwd.contiguous()
    z_gate_bwd,f_gate_bwd = z_gate_bwd.contiguous(),f_gate_bwd.contiguous()
    print((z_gate_fwd-z_gate_bwd).abs().sum().item())
    print((f_gate_fwd-f_gate_bwd).abs().sum().item())


def test_non_inplace():
    linear = nn.Linear(20,6)

    x_fwd = torch.randn(7,1,20)
    x_bwd = x_fwd.clone().flip(1)

    y_fwd = linear(x_fwd)
    y_bwd = linear(x_bwd)
    print((y_fwd-y_bwd).abs().sum().item())
    z_gate_fwd,f_gate_fwd,o_gate_fwd = y_fwd.chunk(3, dim=2)
    z_gate_bwd,f_gate_bwd,o_gate_bwd = y_bwd.chunk(3, dim=2)
    print((z_gate_fwd-z_gate_bwd).abs().sum().item())
    print((f_gate_fwd-f_gate_bwd).abs().sum().item())
    print((o_gate_fwd-o_gate_bwd).abs().sum().item())
    z_gate_fwd = z_gate_fwd.tanh()
    z_gate_bwd = z_gate_bwd.tanh()
    f_gate_fwd = f_gate_fwd.sigmoid()
    f_gate_bwd = f_gate_bwd.sigmoid()
    z_gate_fwd,f_gate_fwd = z_gate_fwd.contiguous(),f_gate_fwd.contiguous()
    z_gate_bwd,f_gate_bwd = z_gate_bwd.contiguous(),f_gate_bwd.contiguous()
    print((z_gate_fwd-z_gate_bwd).abs().sum().item())
    print((f_gate_fwd-f_gate_bwd).abs().sum().item())

0.0
0.0
0.0
0.0
0.0
0.0

0.0
0.0
0.0
0.13790962100028992
0.1261766254901886