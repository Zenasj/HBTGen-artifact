import torch
import torch.nn as nn

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):        
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
            w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()        
        return self.module.forward(*args)

class RNNModel(torch.nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()

        rnn = torch.nn.LSTM(6, 128)
        self.rnn = WeightDrop(rnn, ['weight_hh_l0'], dropout=0.5)


    def forward(self, input_data, return_h=False):
        output, _ = self.rnn(input_data)

        return output


model = RNNModel().cuda()
ones = torch.ones([128, 64, 6]).cuda()
x = model(ones)

model = RNNModel()
ones = torch.ones([128, 64, 6])
x = model(ones)