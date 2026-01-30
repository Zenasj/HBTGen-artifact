import torch
import torch.nn as nn

def repro():
    w = torch.tensor([[-2.6465,  2.5859,  0.4688,  1.7949,  3.2676],
        [-3.1641,  8.9375,  5.7578, -2.9453, -6.5469],
        [ 2.0469,  1.3516, -8.7344,  6.0000,  1.3906],
        [ 6.5781,  7.8438,  6.9766,  3.2891, -5.1172],
        [-7.9414,  7.7344,  4.1875,  2.8574,  2.9531],
        [-0.4844, -5.6328, -6.8359, -4.5156,  3.7891],
        [ 4.9375,  6.6094,  6.7031,  0.6719, -6.4219],
        [ 7.0469,  8.2031,  4.4453,  1.7129, -2.4688],
        [ 1.2207, -3.3750, -2.4531,  7.4062, -6.0469],
        [-8.9688,  2.2656,  2.4160, -1.0176,  8.4531]], dtype=torch.float32, requires_grad=True)
    x = torch.tensor(5)
    out = torch.nn.functional.embedding(x, w)
    out.sum().backward()

    w_mps = w.detach().clone().to("mps").requires_grad_()
    x_mps = x.to("mps")
    out = torch.nn.functional.embedding(x_mps, w_mps)
    out.sum().backward() # error