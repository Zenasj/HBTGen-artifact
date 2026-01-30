import torch
import torch.nn as nn
import numpy as np
import random

imported_encoder = torch.jit.load("path/to/the/model/you/just/unzipped")
imported_encoder.forward(torch.zeros(1, 1, 138, 80, dtype=torch.float))

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5,1), padding=(2,0))
        self.relu_1 = nn.LeakyReLU()
        self.bn_1 = nn.BatchNorm2d(num_features=10) 

        self.conv_2 = nn.Conv2d(in_channels=10, out_channels=1, kernel_size=(5,1), padding=(2,0))
        self.relu_2 = nn.LeakyReLU()
        self.bn_2 = nn.BatchNorm2d(num_features=1)

        self.encoder = nn.GRU(input_size=80, hidden_size=64, batch_first=True, bidirectional=True, num_layers=2)

    def forward(self, spectrogram):
        #Conv block 1
        x = spectrogram
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.bn_1(x)

        #Conv block 2
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.bn_2(x)

        #Encoder
        x = torch.squeeze(x, dim=1)
        encoder_output, enc_h = self.encoder(x)

        #Reshape hidden state to match the dimensions expected by the decoder
        h_layers_twice, h_batch_size, h_output_dim = enc_h.shape
        unidirectional_h = enc_h.reshape((h_layers_twice // 2, h_batch_size, h_output_dim * 2))

        return encoder_output, unidirectional_h

encoder = encoder.eval()
mock_encoder_input = torch.tensor(np.random.rand(1, 1, 138, 80), dtype=torch.float)
traced_encoder = torch.jit.trace(encoder, mock_encoder_input)
traced_encoder.save("path/to/storage")