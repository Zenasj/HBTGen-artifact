import torch.nn as nn

from typing import List, Tuple, Optional, overload, Union, cast
import torch
import numpy as np
import time

class RNNTest():

    def __init__(self, device) -> None:
        pass
    
    def RNNScript(self, input):
        
        outs = []
        in0 = input[0, :]
        dim0 = input.shape[0]

        input_padded = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, 10-dim0))

        outs = []
        for t in range(10):
            def true_fn(carry, inp):
                return torch.sigmoid(inp) + carry
            def false_fn(carry, inp):
                return torch.zeros_like(carry)

            in0 = torch.cond(t < dim0, true_fn, false_fn, (in0, input_padded[t, :]))
            outs.append(in0)

        outs = torch.stack(outs)

        return outs, None

if __name__ == "__main__":

    input_size = 140
    batch_size = 64
    use_gpu = True

    if use_gpu:
        device = torch.device('cuda:0')
    else:
        device = None

    rnn_test = RNNTest(device)

    rnnscript = torch.compile(rnn_test.RNNScript, dynamic=True)
    #rnnscript = rnn_test.RNNScript

    fwd_times = []

    for execution in range(20):
        t_rnd = 8
        inp = torch.rand((t_rnd, batch_size, input_size))
        print(inp.shape)
        if use_gpu:
            inp = inp.cuda()
        start_forward = time.time_ns()
        out, state = rnnscript(inp)
        stop_forward = time.time_ns()
        f_time = (stop_forward - start_forward) / (10 ** 9)
        fwd_times.append(f_time)

    print('Time of forward computation: {:.4f} +- {:.4f} s'.format(np.mean(np.array(fwd_times)),
                                                                   np.std(np.array(fwd_times))))