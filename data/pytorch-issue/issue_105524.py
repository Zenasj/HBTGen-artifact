import torch.nn as nn

from typing import List, Tuple, Optional, overload, Union, cast
import torch
import numpy as np
import time
import torch.optim as optim
from torch.nn.parameter import Parameter

class RNNTest():

    def __init__(self, device) -> None:
        w_ih = torch.empty((input_size, hidden_size), device=device)
        w_io = torch.empty((input_size, hidden_size), device=device)
        self.param1 = torch.cat([w_ih,w_io],1)
        self.param1.requires_grad_(True)

        w_hh = torch.empty((hidden_size, hidden_size), device=device)
        w_ho = torch.empty((hidden_size, hidden_size), device=device)
        self.param2 = torch.cat([w_hh,w_ho],1)
        self.param2.requires_grad_(True)
    
    def RNNScript(self,
        input,
        ):

        state1 = torch.zeros(64, 340, dtype=input.dtype, device=input.device)
        
        outs = []

        Wx = input @ self.param1
        Wx_inp, Wx_rec = torch.tensor_split(Wx, 2, 2)
        for wt_inp, wt_rec in zip(Wx_inp, Wx_rec):
            rec_mul_inp, rec_mul_rec = torch.tensor_split(state1 @ self.param2, 2, 1)
            input_prev = (wt_inp + rec_mul_inp)
            output_gate = (wt_rec + rec_mul_rec)

            state1 = 1. + input_prev * torch.sigmoid(output_gate)
            outs.append(state1)
        
        outs = torch.stack(outs)

        return outs, None

if __name__ == "__main__":

    input_size = 140
    hidden_size = 340
    batch_size = 64
    use_gpu = True

    forward_times = []
    backward_times = []

    if use_gpu:
        device = torch.device('cuda:0')
    else:
        device = None

    rnn_test = RNNTest(device)
    
    def count_kernels(guard):
        print("[pt2_compile] guard failed: ", guard)

    #rnnscript = torch.compile(rnn_test.RNNScript, mode='reduce-overhead', dynamic=False, fullgraph=True)
    rnnscript = torch.compile(rnn_test.RNNScript, mode='reduce-overhead', dynamic=True, fullgraph=False)
    #rnnscript = torch.compile(rnn_test.RNNScript, dynamic=True, fullgraph=True)
    
    optimizer = optim.SGD([rnn_test.param1, rnn_test.param2], 0.1)

    optimizer.zero_grad()
    for execution in range(20):
        start_forward = time.time_ns()
        t_rnd = 120
        inp = torch.rand((t_rnd, batch_size, input_size))
        print(inp.shape)
        if use_gpu:
            inp = inp.cuda()
        
        out, state = rnnscript(inp)

        if use_gpu:
            torch.cuda.synchronize()
        stop_forward = time.time_ns()
        forward_times.append((stop_forward - start_forward) / (10 ** 9))
        
        loss = 1. - torch.sum(out)

        start_time_backward = time.time_ns()
        loss.backward()
        if use_gpu:
            torch.cuda.synchronize()
        stop_time_backward = time.time_ns()
        backward_times.append((stop_time_backward - start_time_backward) / (10 ** 9))