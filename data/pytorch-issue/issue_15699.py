# torch.rand(B, 3, 256, 256, dtype=torch.float32)
import torch
import torch.nn as nn
import functools

class MyModel(nn.Module):
    class ResnetBlock(nn.Module):
        def __init__(self, dim, norm_layer, use_dropout, use_bias):
            super(MyModel.ResnetBlock, self).__init__()
            block = [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
                norm_layer(dim),
                nn.ReLU(True)
            ]
            if use_dropout:
                block += [nn.Dropout(0.5)]
            block += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
                norm_layer(dim)
            ]
            self.block = nn.Sequential(*block)

        def forward(self, x):
            return x + self.block(x)

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9):
        super(MyModel, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
                MyModel.ResnetBlock(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                  kernel_size=3, stride=2,
                                  padding=1, output_padding=1,
                                  bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

def my_model_function():
    return MyModel(input_nc=3, output_nc=3, n_blocks=9, norm_layer=nn.InstanceNorm2d)

def GetInput():
    return torch.rand(1, 3, 256, 256, dtype=torch.float32)

