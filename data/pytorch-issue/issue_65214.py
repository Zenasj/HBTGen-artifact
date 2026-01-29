# torch.rand(B, 100, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class G(nn.Module):
    feature_maps = 512
    kernel_size = 4
    stride = 2
    padding = 1
    bias = True

    def __init__(self, input_vector, ngpu=0):
        super(G, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_vector, self.feature_maps, self.kernel_size, 1, 0, bias=self.bias),
            nn.BatchNorm2d(self.feature_maps), nn.ReLU(True),
            nn.ConvTranspose2d(self.feature_maps, self.feature_maps // 2, self.kernel_size, self.stride, self.padding,
                               bias=self.bias),
            nn.BatchNorm2d(self.feature_maps // 2), nn.ReLU(True),
            nn.ConvTranspose2d(self.feature_maps // 2, (self.feature_maps // 2) // 2, self.kernel_size, self.stride,
                               self.padding,
                               bias=self.bias),
            nn.BatchNorm2d((self.feature_maps // 2) // 2), nn.ReLU(True),
            nn.ConvTranspose2d((self.feature_maps // 2) // 2, ((self.feature_maps // 2) // 2) // 2, self.kernel_size,
                               self.stride, self.padding,
                               bias=self.bias),
            nn.BatchNorm2d(((self.feature_maps // 2) // 2) // 2), nn.ReLU(True),
            nn.ConvTranspose2d(((self.feature_maps // 2) // 2) // 2, 4, self.kernel_size, self.stride, self.padding,
                               bias=self.bias),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class D(nn.Module):
    feature_maps = 64
    kernel_size = 4
    stride = 2
    padding = 1
    bias = True
    inplace = True

    def __init__(self, ngpu=0):
        super(D, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(4, self.feature_maps, self.kernel_size, self.stride, self.padding, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=self.inplace),
            nn.Conv2d(self.feature_maps, self.feature_maps * 2, self.kernel_size, self.stride, self.padding,
                      bias=self.bias),
            nn.BatchNorm2d(self.feature_maps * 2), nn.LeakyReLU(0.2, inplace=self.inplace),
            nn.Conv2d(self.feature_maps * 2, self.feature_maps * 4, self.kernel_size, self.stride, self.padding,
                      bias=self.bias),
            nn.BatchNorm2d(self.feature_maps * 4), nn.LeakyReLU(0.2, inplace=self.inplace),
            nn.Conv2d(self.feature_maps * 4, self.feature_maps * 8, self.kernel_size, self.stride, self.padding,
                      bias=self.bias),
            nn.BatchNorm2d(self.feature_maps * 8), nn.LeakyReLU(0.2, inplace=self.inplace),
            nn.Conv2d(self.feature_maps * 8, 1, self.kernel_size, 1, 0, bias=self.bias),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

class MyModel(nn.Module):
    def __init__(self, input_vector=100, ngpu=0):
        super(MyModel, self).__init__()
        self.generator = G(input_vector, ngpu)
        self.discriminator = D(ngpu)
        self.apply(weights_init)  # Initialize weights

    def forward(self, input):
        gen_image = self.generator(input)
        disc_out = self.discriminator(gen_image)
        return disc_out  # Output discriminator's decision on generated image

def my_model_function():
    return MyModel(input_vector=100)  # Matches GAN's input_vector=100

def GetInput():
    return torch.rand(64, 100, 1, 1, dtype=torch.float32)  # Noise vector for generator

