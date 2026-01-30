import torch
import torch.nn as nn

class BasicUnit(nn.Module):
    """
    The basic recurrent unit for the vanilla stacked RNNs.

    Parameters
    ----------
    unit : ``torch.nn.Module``, required.
        The type of rnn unit.
    input_dim : ``int``, required.
        The input dimension fo the unit.
    hid_dim : ``int``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    batch_norm: ``bool``, required.
        Incorporate batch norm or not. 
    """
    def __init__(self, unit, input_dim, hid_dim, droprate, batch_norm):
        super(BasicUnit, self).__init__()

        self.unit_type = unit
        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        self.layer = nn.LSTM(input_dim, hid_dim//2, 1, batch_first=True, bidirectional=True)#need consider
        
        self.droprate = droprate
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(hid_dim)
        self.output_dim = hid_dim

class BasicRNN(nn.Module):
    """
    The multi-layer recurrent networks for the vanilla stacked RNNs.

    Parameters
    ----------
    layer_num: ``int``, required.
        The number of layers. 
    unit : ``torch.nn.Module``, required.
        The type of rnn unit.
    input_dim : ``int``, required.
        The input dimension fo the unit.
    hid_dim : ``int``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    batch_norm: ``bool``, required.
        Incorporate batch norm or not. 
    """
    def __init__(self, layer_num, unit, emb_dim, hid_dim, droprate, batch_norm):
        super(BasicRNN, self).__init__()

        self.layer_list = [BasicUnit(unit, emb_dim, hid_dim, droprate, batch_norm)] + [BasicUnit(unit, hid_dim, hid_dim, droprate, batch_norm) for i in range(layer_num - 1)]
        self.layer = nn.Sequential(*self.layer_list)
        self.output_dim = self.layer_list[-1].output_dim

        self.init_hidden()