import torch.nn as nn

py
import torch

torch.manual_seed(420)

batch_size = 2
input_tensor = torch.randn((batch_size, 10))

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.idx = torch.tensor([[1, 3, 2],[1, 4, 3]])

    def forward(self, x):
        return torch.sum(x[self.idx], dim=1)

func = Model().to('cpu')
test_inputs = [input_tensor]


with torch.no_grad():
    func.train(False)

    jit_func = torch.compile(func)
    res2 = jit_func(input_tensor)
    print(res2)
    # success, out-of-bound read
    # tensor([[ 6.1276e-01, -9.3005e-01,  1.3646e+00, -7.3723e-01, -7.0839e-01,
    #     -2.8423e-01, -1.4816e+00,  3.2976e-01,  4.8557e-01,  4.1309e-01],
    #    [ 6.1276e-01, -9.3005e-01,  1.3646e+00,         nan, -7.0839e-01,
    #     -1.7412e+38, -1.4816e+00,         nan,  4.8557e-01,  4.1309e-01]])

    func(input_tensor) # without jit
    # IndexError: index 3 is out of bounds for dimension 0 with size 2