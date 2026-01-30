import torch.nn as nn
import numpy as np

import torch

def test_lstm(num_layers, bidirectional, batch_first):
    torch.manual_seed(1234)
    lstm = torch.nn.LSTM(5, 5, num_layers=num_layers, bidirectional=bidirectional, batch_first=batch_first)
    lstm.eval()
    inp = torch.randn(4, 1, 5)
    if batch_first:
        inp = inp.transpose(0, 1)
    print("num_layers {} bidirectional {} batch_first {}".format(num_layers, bidirectional, batch_first), end="  ")
    print(torch.linalg.norm(lstm(inp)[0]).item(), end="  ")
    print(torch.linalg.norm(lstm.to("mps")(inp.to("mps"))[0]).item())

test_lstm(2, True, True)   # bad
test_lstm(2, True, False)  # bad
test_lstm(1, True, True)   # bad

test_lstm(2, False, True)  # ok
test_lstm(1, False, True)  # ok
test_lstm(1, False, False) # ok

import torch

def test_lstm(num_layers, bidirectional, batch_first):
    torch.manual_seed(1234)
    lstm = torch.nn.LSTM(5, 5, num_layers=num_layers, bidirectional=bidirectional, batch_first=batch_first)
    lstm.eval()
    inp = torch.randn(4, 2, 5)   # the only change I made was to make a batch of size 2
    if batch_first:
        inp = inp.transpose(0, 1)
    print("num_layers {} bidirectional {} batch_first {}".format(num_layers, bidirectional, batch_first), end="  ")
    print(torch.linalg.norm(lstm(inp)[0]).item(), end="  ")
    print(torch.linalg.norm(lstm.to("mps")(inp.to("mps"))[0]).item())

test_lstm(2, True, True)   # bad                                                                                                                                                     
test_lstm(2, True, False)  # bad                                                                                                                                                     
test_lstm(1, True, True)   # bad                                                                                                                                                     

test_lstm(2, False, True)  # ok                                                                                                                                                      
test_lstm(1, False, True)  # ok                                                                                                                                                      
test_lstm(1, False, False) # ok