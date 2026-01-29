# torch.rand(B, C, H, W, dtype=...)  # Input shape: (batch_size, 3, 32, 32)
import torch
import torch.nn as nn

class MeanOnlyBatchNorm(nn.Module):
    def __init__(self, module, momentum=0.1):
        super(MeanOnlyBatchNorm, self).__init__()
        if isinstance(module, nn.Linear):
            self.num_features = module.out_features
        else:
            self.num_features = module.out_channels
        self.do_init = False
        self.momentum = momentum
        self.weight = nn.Parameter(torch.Tensor(self.num_features))
        self.bias = nn.Parameter(torch.Tensor(self.num_features))
        self.register_buffer('running_mean', torch.zeros(self.num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        self.do_init = False

    def forward(self, x):
        exp_dim = (None, slice(None)) + tuple(None for _ in range(2, x.dim()))
        if self.training:
            red_dim = (0,) + tuple(range(2, x.dim()))
            mean = x.mean(red_dim)
            activation = x - mean[exp_dim]
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
        else:
            activation = x - self.running_mean[exp_dim]

        return activation + self.bias[exp_dim]

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine=True, ' \
               'track_running_stats=True'.format(**self.__dict__)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        n_class = 10
        c1 = nn.Conv2d(3, 1000, kernel_size=3, padding=1, stride=4)
        bn1 = MeanOnlyBatchNorm(c1)
        act1 = nn.ReLU(True)
        pool1 = nn.AdaptiveAvgPool2d(1)
        n2 = nn.Conv2d(1000, n_class, kernel_size=1)
        bn2 = MeanOnlyBatchNorm(n2)
        self.model = nn.Sequential(c1, bn1, act1, pool1, n2, bn2)

    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 100
    input_size = (3, 32, 32)
    device = torch.device('cuda')
    return torch.randn(*((batch_size,) + input_size), device=device)

