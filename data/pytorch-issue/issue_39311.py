import torch.nn as nn

def lstm_spectral_norm(self,input_size, hidden_size, n_layers, batch_first=True, bidirectional = False):
        lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first, bidirectional)
        name_pre = 'weight'
        for i in range(n_layers):
            name = name_pre+'_hh_l'+str(i)
            spectral_norm(lstm, name)
            name = name_pre+'_ih_l'+str(i)
            spectral_norm(lstm, name)
        return lstm

import torch

def lstm_spectral_norm(input_size, hidden_size, n_layers=1):
    lstm = torch.nn.LSTM(input_size, hidden_size, n_layers)
    name_pre = 'weight'
    for i in range(n_layers):
        name = name_pre+'_hh_l'+str(i)
        torch.nn.utils.spectral_norm(lstm, name)
        name = name_pre+'_ih_l'+str(i)
        torch.nn.utils.spectral_norm(lstm, name)
    return lstm

ninp = 128
m = lstm_spectral_norm(ninp, ninp)
name = 'weight_hh_l0'
m = m.cuda()
print([(name, p.device) for name, p in m.named_parameters()])
inp = torch.randn(3, 32, 128, device="cuda")