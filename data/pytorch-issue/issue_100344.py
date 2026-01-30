import torch.nn as nn

py
import torch

torch.manual_seed(420)

input_tensor = torch.randn((1, 3, 10, 10))

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_transpose(x, output_size=(10, 10))
        return x

func = Model().to('cpu')
test_inputs = [input_tensor]


with torch.no_grad():
    func.train(False)
    res1 = func(input_tensor) # without jit
    print(res1)
    # success

    jit_func = torch.compile(func)
    res2 = jit_func(input_tensor)
    print(res2)
    # TypeError: forward() got an unexpected keyword argument 'output_size'
    # While executing %l__self___conv_transpose : [#users=1] = call_module[target=L__self___conv_transpose](args = (%l_x_,), kwargs = {output_size: (10, 10)})