import torch.nn as nn

def __init__(self, input_channels, n_segment=8, n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        input_channels = input_channels
        self.n_segment = n_segment
        fold_div = n_div
        fold = input_channels // fold_div
        self.conv_shift = nn.Conv1d(
            input_channels, input_channels,
            kernel_size=3, padding=1, groups=input_channels,
            bias=False)
        
def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute([0, 3, 4, 2, 1])  # (n_batch, h, w, c, n_segment)
        x = x.contiguous().view(n_batch * h * w, c, self.n_segment)
        
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x

def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute([0, 3, 4, 2, 1]) 
        x = x.contiguous().view(n_batch * h * w, c, self.n_segment)
        x = self.conv_shift(x)  # (n_batch*h*w, c, n_segment)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute([0, 4, 3, 1, 2])  # (n_batch, n_segment, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x