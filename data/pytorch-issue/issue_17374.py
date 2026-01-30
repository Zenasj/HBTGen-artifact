from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.jit import ScriptModule, script_method, trace

# Comment(david.dai): nn.Module --> torch.jit.ScriptModule. Use ScriptModule
# instead of trace due to for loops.
#class Sequence(nn.Module):
class Sequence(torch.jit.ScriptModule):
    def __init__(self, batch_size):
        super(Sequence, self).__init__()
        self.b = batch_size
        self.x_dim = 1
        self.h_dim = 3
        # Trace static modules.
        # Comment(david.dai): The trace unfortunately fixes the batch size for both
        # train and serving time. It's still an active area of development as of
        # 2/12/2019. Related issues
        # https://github.com/pytorch/pytorch/issues/16663
        # https://github.com/pytorch/pytorch/issues/15319
        self.lstm1 = trace(nn.LSTMCell(self.x_dim, self.h_dim),
             (torch.rand(self.b, self.x_dim), (torch.rand(self.b, self.h_dim),
                 torch.rand(self.b, self.h_dim))))
        self.lstm2 = trace(nn.LSTMCell(self.h_dim, self.h_dim),
             (torch.rand(self.b, self.h_dim), (torch.rand(self.b, self.h_dim),
                 torch.rand(self.b, self.h_dim))))
        self.linear = trace(nn.Linear(self.h_dim, self.x_dim),
             (torch.rand(self.b, self.h_dim),))
        #self.lstm1 = nn.LSTMCell(1, 51)
        #self.lstm2 = nn.LSTMCell(51, 51)
        #self.linear = nn.Linear(51, 1)

    # Comment(david.dai): Declare constant member variables.
    __constants__ = ["b", "h_dim", "x_dim"]

    @torch.jit.script_method
    def forward(self, input : torch.Tensor, future : int):
    #def forward(self, input, future=10):
        # Python list isn't supported in JIT.
        #outputs = []
        outputs = torch.zeros(size=(self.b, 0), dtype=torch.double)
        h_t = torch.zeros(input.size(0), self.h_dim, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.h_dim, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), self.h_dim, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), self.h_dim, dtype=torch.double)

        # Comment(david.dai):
        # "Script currently does not support iterating over generic iterable
        # objects like lists or tensors. Script currently does not support
        # start or increment parameters to range. These will be added in a
        # future version." (https://pytorch.org/docs/stable/jit.html)
        #for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):

        # Comment(david.dai): 'output', if declared within for loop, is not
        # visible outside of the for loop (i.e., invisible to the second for
        # loop)
        output = torch.zeros(self.b, self.x_dim, dtype=torch.double)
        for i in range(input.size(1)):
            input_t = input[:, i].unsqueeze(1)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)

            # Comment(david.dai): This list concatenation is interpreted as
            # tensor addition (aten::add) and causes # Torch JIT compilation
            # error. Convert into torch tensor.
            #outputs += [output]
            outputs = torch.cat((outputs, output), dim=1)

        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            # Comment(david.dai): Same as above
            #outputs += [output]
            outputs = torch.cat((outputs, output), dim=1)

        # Comment(david.dai): outputs is already torch.Tensor.
        #outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


data = torch.load('traindata.pt')
input = torch.from_numpy(data[3:, :-1])
target = torch.from_numpy(data[3:, 1:])

scripted_seq = Sequence(batch_size=2)
torch.onnx._export(scripted_seq, (input, 4), "rnn.onnx", verbose=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        example_outputs=target)

torch.onnx._export(scripted_seq, (input, torch.tensor(4)), "rnn.onnx", verbose=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        example_outputs=target)