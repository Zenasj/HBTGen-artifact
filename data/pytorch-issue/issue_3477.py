import torch

import torch.nn as nn
from torch.autograd import Variable

seq_len = 10
features = 50
hidden_size = 50
batch_size = 32

model = nn.Module()
model.rnn = nn.RNN(input_size=features, hidden_size=hidden_size, num_layers=2)
model.cuda(5)

X_train = torch.randn(seq_len, batch_size, features)
y_train = torch.randn(batch_size)
X_train, y_train = Variable(X_train).cuda(), Variable(y_train).cuda()

class device(object):
    """Context-manager that changes the selected device.

    Arguments:
        idx (int): device index to select. It's a no-op if this argument
            is negative.
    """

    def __init__(self, idx):
        self.idx = idx
        self.prev_idx = -1

    def __enter__(self):
        if self.idx is -1:
            return
        _lazy_init()
        self.prev_idx = torch._C._cuda_getDevice()
        if self.prev_idx != self.idx:
            torch._C._cuda_setDevice(self.idx)

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            torch._C._cuda_setDevice(self.prev_idx)
        return False

model = nn.Module()
model.rnn = nn.RNN(input_size=features, hidden_size=hidden_size, num_layers=2)
model.cuda(5)

X_train = torch.randn(seq_len, batch_size, features)
y_train = torch.randn(batch_size)
X_train, y_train = Variable(X_train).cuda(5), Variable(y_train).cuda(5)