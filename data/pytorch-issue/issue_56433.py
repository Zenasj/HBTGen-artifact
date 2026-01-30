import torch
import torch.nn as nn

rnn = nn.LSTM(512, 512, 2, dropout=0.5).cuda()

out1 = rnn(in1)

# calling cudnn rnn with dropout in capture after calling it uncaptured triggers 1
capture_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(capture_stream):
    graph.capture_begin()
    out2 = rnn(in2)
    graph.capture_end()
torch.cuda.current_stream().wait_stream(capture_stream)

# calling cudnn rnn with dropout uncaptured after calling it in capture triggers 2
out3 = rnn(in3)