import torch
import torch.nn as nn

with torch.set_grad_enabled(False):
    pred1 = model(x)
    
with torch.set_grad_enabled(True):
    pred2 = model(x)


print(
        (nn.functional.l1_loss(y, pred1),
         nn.functional.l1_loss(pred1, y),
         nn.functional.l1_loss(y, pred2), 
         nn.functional.l1_loss(pred2, y))
    )

def get_conv(in_channels, out_channels, kernel_size=3, actn=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)]
    if actn: layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)
    

class ResSequential(nn.Module):
    def __init__(self, layers, mult):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.mult = mult
    
    def forward(self, input):
        return input + self.layers(input) * self.mult

def res_block(num_features):
    layers = [get_conv(num_features, num_features),
              get_conv(num_features, num_features, actn=False)]
    return ResSequential(layers, 0.1)

def upsample(in_channels, out_channels, scale):
    layers = []
    for i in range(int(log(scale, 2))):
        layers += [get_conv(in_channels, out_channels * 4), nn.PixelShuffle(2)]
        
    return nn.Sequential(*layers)

class SuperResNet(nn.Module):
    def __init__(self, scale, nf=64):
        super().__init__()
        
        layers = [
            get_conv(3, nf),
            *[res_block(nf) for i in range(8)],
            upsample(nf, nf, scale),
            nn.BatchNorm2d(nf),
            get_conv(nf, 3, actn=False),
        ]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

model = SuperResNet(scale)

device = torch.device("cuda:0")
model = model.to(device)

model

def safe_loss(f):
    """
    When loss is decorated with a `safe_loss`, it helps you avoid a bug with incorrect arguments order.
    """

    @wraps(f)
    def wrapper(y_pred, y_true, **kwargs):
        if y_true.grad_fn is not None:
            warn('Usually y_true should have no gradients attached. Please make sure you\'re calling the loss properly')
        return f(y_pred, y_true, **kwargs)

    return wrapper