import numpy
import torch
import torch.nn as nn
import torch.jit as jit


class MyLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(MyLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state

        gates = self.weight_ih(input) + self.weight_hh(hx)
        
        ingate, forgetgate, cellgate, outgate = torch.split(gates, self.hidden_size)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hx, (hy, cy)

class LoopModel(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(LoopModel, self).__init__()
        self.lstm = MyLSTMCell(input_size, hidden_size)

    @jit.script_method
    def forward(self, x):
        # type: (Tensor)-> (Tensor)
        hx = torch.randn(20)
        cx = torch.randn(20)
        output = []
        for i in range(len(x)):
            a, (hx, cx) = self.lstm(x[i], (hx, cx))
            output.append(hx)
        return hx


def generate_model():
    model = LoopModel(8, 20)
    dummy_input = torch.ones(3, 8, dtype=torch.float)
    o = model(dummy_input)
    torch.onnx.export(model, dummy_input, 'loop.onnx', verbose=True,
                    input_names=['input_data'], example_outputs=o)
    print("done")

generate_model()