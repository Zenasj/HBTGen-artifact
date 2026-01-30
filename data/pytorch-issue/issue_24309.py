import torch.nn as nn
import torch
def make_conv(in_dim, out_dim, kernel_size=(3, 3), stride=1,
              padding=1, activation=None, norm_type=''):
    layer = []
    if norm_type == 'SN':
        layer += [nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding))]
    elif norm_type in ('BN', 'synBN'):
        layer += [nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
                  nn.BatchNorm2d(out_dim, momentum=0.8, eps=1e-3)]
    elif norm_type == 'IN':
        layer += [nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
                  nn.InstanceNorm2d(out_dim, affine=False, track_running_stats=False)]
    else:
        layer += [nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)]
    if activation is not None:
        layer += [activation]
    return nn.Sequential(*layer)
class AdShuffleGenerator(nn.Module):
    def __init__(self, n1=64, n2=64, n3=64, f1=1, f2=4, f3=1, n_rn=9, updown_conv=-1, activation='prelu',
                 activation_final='tanh', norm_type='SN', senet=False):
        super().__init__()
        blur_upsample_flag = False
        if updown_conv < 0:
            updown_conv = int(-updown_conv)
            blur_upsample_flag = True
        if activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        else:
            self.activation = nn.ReLU()
        if activation_final == 'tanh':
            self.activation_final = nn.Tanh()
        elif activation_final == 'prelu':
            self.activation_final = nn.PReLU()
        elif activation_final == 'sigmoid':
            self.activation_final = nn.Sigmoid()
        else:
            self.activation_final = None
        k1_padding = int(f1 / 2) if f1 % 2 and f1 > 1 else 0
        k3_padding = int(f3 / 2) if f3 % 2 and f3 > 1 else 0
        self.layer0 = make_conv(3, n1, kernel_size=f1, padding=k1_padding, activation=self.activation,
                                norm_type=norm_type)
        layer_down = []
        for i in range(0, updown_conv):
            layer_down += [make_conv(n1, n2, kernel_size=f2, stride=2, activation=self.activation, norm_type=norm_type)]
        self.layer_down = nn.Sequential(*layer_down)
        layer_up = []
        for i in range(0, updown_conv):
            layer_up += [make_conv(n2, n2 * 4, activation=self.activation, norm_type=norm_type), nn.PixelShuffle(2)]
            if blur_upsample_flag:
                layer_up += [nn.ReplicationPad2d((1, 0, 1, 0)), nn.AvgPool2d(2, stride=1)]
        self.layer_up = nn.Sequential(*layer_up)
        self.final_layer = nn.Conv2d(n2, 3, kernel_size=f3, padding=k3_padding)
    def forward(self, x):
        x = self.layer0(x)
        x = self.layer_down(x)
        x = self.layer_up(x)
        x = self.final_layer(x)
        if self.activation_final is not None:
            x = self.activation_final(x)
        return x
net = AdShuffleGenerator().cuda()
a = torch.rand(1,3, 5761, 3841).cuda()
with torch.no_grad():
    y_pred = net.layer0(a)
    y_pred = net.layer_down(y_pred)
    y_pred = net.layer_up[0](y_pred)
    y_pred = net.layer_up[1](y_pred)
    y_pred = net.layer_up[2](y_pred)
print(y_pred.max())
torch.cuda.empty_cache()
with torch.no_grad():
    y_pred = net.layer_up[3](y_pred)
print(y_pred.max())

from fastai.layers import PixelShuffle_ICNR
class PixelShuffle_ICNR(nn.Module):
    def __init__(self, ni:int, nf:int=None, scale:int=2, blur:bool=False, norm_type=NormType.Weight, leaky:float=None):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = conv_layer(ni, nf*(scale**2), ks=1, norm_type=norm_type, use_activ=False)
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1,0,1,0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = relu(True, leaky=leaky)
    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x