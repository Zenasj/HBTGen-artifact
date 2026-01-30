import torch
import torch.nn as nn
from Common.Src.activation_funciton import get_activation_func

class Conv2D(nn.Module):
    def __init__(self, 
                 n_in_channel : int, 
                 n_out_channel : int, 
                 kernel, # int or (int, int)
                 stride, # int or (int, int)
                 padding, # int or (int, int) or str("valid" or "same")
                 n_layer : int = 1, 
                 on_batch_norm : bool = True, 
                 activation_func : str = "relu"):
        assert type(kernel) is int or type(kernel) is (int, int)
        assert type(stride) is int or type(stride) is (int, int)
        assert type(padding) is int or type(padding) is (int, int) or (type(padding) is str and str.lower(padding) in ["valid", "same"])

        super(Conv2D, self).__init__()
        self.n_in_channel = n_in_channel
        self.n_out_channel = n_out_channel
        self.n_layer = n_layer
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.on_batch_norm = on_batch_norm
        self.batch_norms = []
        for _ in range(n_layer):
            self.batch_norms.append(nn.BatchNorm2d(n_out_channel))

        self.conv2ds = [ nn.Conv2d(self.n_in_channel, self.n_out_channel, self.kernel, self.stride, self.padding)]
        for i in range(n_layer - 1):
            self.conv2ds.append(nn.Conv2d(self.n_out_channel, self.n_out_channel, self.kernel, self.stride, self.padding))

        self.activation_funcs = []
        for _ in range(n_layer):
            self.activation_funcs.append(get_activation_func(activation_func))

        self.seq = nn.Sequential()
        for i in range(n_layer):
            self.seq.append(self.conv2ds[i])
            if self.on_batch_norm:
                self.seq.append(self.batch_norms[i])
            self.seq.append(self.activation_funcs[i])

    def forward(self, x : torch.Tensor):
        x = self.seq(x)
        # for i in range(self.n_layer):
        #     x = self.conv2ds[i](x)
        #     if self.on_batch_norm:
        #         x = self.batch_norms[i](x)
        #     x = self.activation_funcs[i](x)
        
        return x

net = PaddedUNet(3, 3).to(device)

x, y = x.to(device), y.to(device)

import torch
import torch.nn as nn

class TNet(nn.Module):
    def __init__(self, k : int):
        super(TNet, self).__init__()

        self.k = k
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.s0 = nn.Conv1d(k, 64, 1)
        self.s1 = nn.BatchNorm1d(64)
        self.s2 = nn.ReLU()

        self.s3 = nn.Conv1d(64, 128, 1)
        self.s4 = nn.BatchNorm1d(128)
        self.s5 = nn.ReLU()

        self.s6 = nn.Conv1d(128, 1024, 1)
        self.s7 = nn.BatchNorm1d(1024)
        self.s8 = nn.ReLU()

        self.s_list = [self.s0, self.s1, self.s2, self.s3, self.s4, self.s5, self.s6, self.s7, self.s8 ]
        self.s_list2 = [nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.ReLU(), 
                        nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                        nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU() ]

        self.fc_network = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, self.k * self.k)
        )
    
    def forward(self, x):
        batch_size : int = x.size()[0]

        # 1. Sequantial
        # x = self.shared_mlp(x)

        # 2. Chaining of attributes
        # x = self.s0(x)
        # x = self.s1(x)
        # x = self.s2(x)
        # x = self.s3(x)
        # x = self.s4(x)
        # x = self.s5(x)
        # x = self.s6(x)
        # x = self.s7(x)
        # x = self.s8(x)

        # 3. List of attributes
        # for i in range(len(self.s_list)):
        #     x = self.s_list[i](x)

        # 4. List of Modules.
        # Only this raised an error.
        for i in range(len(self.s_list2)):
            x = self.s_list2[i](x)
        
        x = torch.max(x, 2, keepdim=True)[0] # element-wise maximum. (ref. from Theorem 1. on PointNet paper.)
        x = x.view(-1, 1024)
        x = self.fc_network(x) # (, k * k)
        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x