import torch
import torch.nn as nn
a = torch.randn(5, 3, 10) # This can be sent into GPU correctly!
rnn = nn.RNN(10, 20, 2)
rnn.cuda()

model = resnet101(sample_size=input_size, sample_duration=sample_duration)