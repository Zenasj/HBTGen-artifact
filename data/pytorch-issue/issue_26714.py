import torch
import torch.nn as nn
import torch.nn.functional as F

class network (nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.sn_rnn = nn.utils.spectral_norm(nn.RNN(20,10,1),name='weight_hh_l0')
        # self.fc = nn.utils.spectral_norm(nn.Linear(20,10), name='weight')
    def forward (self,x):
        # x = F.tanh(self.fc(x))
        out,_ = self.sn_rnn(x)
        return F.log_softmax(out, dim=1)

if __name__ == '__main__':
    x = torch.randn(2, 10, 20).cuda()
    model = network().cuda()
    out= model(x)
    print('end')