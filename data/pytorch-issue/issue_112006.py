import torch.nn as nn

import torch
def run_on_device():
    tensor = torch.randn([2, 2, 3], requires_grad=True)
    m = torch.nn.GELU(approximate="none")
    # forward result
    res = m(tensor)
    # bwd_tensor
    grad_in = torch.arange(12).reshape([2, 2, 3]).permute([1, 0, 2])
    res.backward(grad_in)
    print(tensor.grad)

if __name__ == "__main__":
    run_on_device()

tensor([[[2.7357e-34, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00]],

        [[0.0000e+00, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 0.0000e+00, 0.0000e+00]]])

tensor([[[ 0.0000, -0.0524,  1.4309],
         [ 5.5390, -0.3139,  0.7154]],

        [[ 2.0605,  4.1351,  5.6143],
         [10.1405, -0.0673, -0.2192]]])