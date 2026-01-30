import torch
from torch.nn import Parameter

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda: mask = mask.cuda()
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)

            # This is because we may call this function in non-training mode first and so, as self.training=False, w is
            # a nn.Parameter and thus self.module.weight remains a Parameter of self.module when we don't want it to.
            if name_w in self.module._parameters:
                del self.module._parameters[name_w]
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

import torch.nn as nn

input_size, batch_size, hidden_size, seq_len = 5, 20, 100, 50
drop_prob = 0.5

device = torch.device('cuda')

model = nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, dropout=0).to(device)
model_wd = WeightDrop(model, ['weight_hh_l0'], dropout=drop_prob)

inputs = torch.randn(seq_len, batch_size, input_size).to(device)
hidden_and_cell = (torch.zeros(1, batch_size, hidden_size, device=device), torch.zeros(1, batch_size, hidden_size, device=device))

model_wd.train()

_ = model_wd(inputs, hidden_and_cell)
### /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:1269: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters().

def forward(self, *args):
        self._setweights()
        self.module.flatten_parameters()
        return self.module.forward(*args)