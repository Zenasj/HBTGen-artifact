import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nfunc
def call_bn(bn, x, update_batch_stats=True):
    if bn.training is False:
        return bn(x)
    elif not update_batch_stats:
        return nfunc.batch_norm(x, None, None, bn.weight, bn.bias, True, bn.momentum, bn.eps)
    else:
        return bn(x)
    
    
class MLP(nn.Module):
    def __init__(self, layer_sizes, affine=False, top_bn=True):
        super(MLP, self).__init__()
        self.input_len = 1 * 28 * 28
        self.fc1 = nn.Linear(self.input_len, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)

        self.bn_fc1 = nn.BatchNorm1d(1200, affine=affine)
        self.bn_fc2 = nn.BatchNorm1d(1200, affine=affine)
        self.top_bn = top_bn
        if top_bn:
            self.bn_fc3 = nn.BatchNorm1d(10, affine=affine)

    def forward(self, x, update_batch_stats=True):
        h = nfunc.relu(call_bn(self.bn_fc1, self.fc1(x.view(-1, self.input_len)), update_batch_stats))
        h = nfunc.relu(call_bn(self.bn_fc2, self.fc2(h), update_batch_stats))
        if self.top_bn:
            h = call_bn(self.bn_fc3, self.fc3(h), update_batch_stats)
        else:
            h = self.fc3(h)
        logits = h
        return logits

model = MLP(None)
a = model(torch.zeros(2, 1, 28, 28))  # it works fine

dummy_input = (torch.zeros(2, 1, 28, 28),)
writer.add_graph(MLP(None), dummy_input, True)

#this fails
out=in_tensor/20 

#this works
out=(in_tensor/20).type(torch.float32)