import torch.nn as nn

class Conv1dSamePad(nn.Module):
    def __init__(self, in_channels, out_channels, filter_len, **kwargs):
        super(Conv1dSamePad, self).__init__()
        self.filter_len = filter_len
        self.conv = nn.Conv1d(in_channels, out_channels, filter_len, padding=(self.filter_len // 2), **kwargs)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        if self.filter_len % 2 == 1:
            return self.conv(x)
        else:
            return self.conv(x)[:, :, :-1]

class ReproduceNet(nn.Module):
    def __init__(self, num_classes):
        super(ReproduceNet, self).__init__()
        self.num_classes = num_classes
        self.first_conv = Conv1dSamePad(13, 1000, 13)
        self.convs = nn.ModuleList(
            [nn.Sequential(Conv1dSamePad(1000, 1000, idx + 10), nn.BatchNorm1d(1000),
                           nn.ReLU(inplace=True)) for idx in range(15)])
        self.last_conv = nn.Sequential(Conv1dSamePad(1000, num_classes, 1))
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.first_conv(x)
        for idx, conv in enumerate(self.convs):
            x = conv(x) 
        return self.logsoftmax(self.last_conv(x))