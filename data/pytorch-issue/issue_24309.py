# torch.rand(1, 3, 5761, 3841, dtype=torch.float32)
import torch
import torch.nn as nn

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

class MyModel(nn.Module):
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

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 5761, 3841, dtype=torch.float32).cuda()

