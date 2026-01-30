import torch
import torch.nn as nn

print(torch.__version__)

class test(torch.jit.ScriptModule):

    def __init__(self, vocab_size=10, rnn_dims=512):
        super().__init__()
        self.word_embeds = nn.Embedding(vocab_size, rnn_dims)
        self.emb_drop = nn.Dropout(0.1)
        self.rnn = nn.LSTM(input_size=rnn_dims, hidden_size=rnn_dims, batch_first=True,
                           num_layers=2, dropout=0.1)
        # delattr(self.rnn, 'forward_packed')

    @torch.jit.script_method
    def forward(self, x):
        h1 = (torch.zeros(2, 1, 512), torch.zeros(2, 1, 512))
        embeds = self.emb_drop(self.word_embeds(x))
        out, h1 = self.rnn(embeds, h1)

        return h1


model = test()

input = torch.ones((1,3)).long()
output = model(input)
torch.onnx.export(model,  # model being run
                  input,
                  'test.onnx',
                  example_outputs=output)

import torch.nn as nn

class test(nn.Module):
    def __init__(self, rnn_dims=32):
        super().__init__()
        self.rnn = nn.LSTM(input_size=rnn_dims, hidden_size=rnn_dims)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out


model = test()
model = model.eval()

input = torch.ones((32, 32, 32)).float()
output = model(input)

torch.onnx.export(model,  # model being run
                  input,
                  'test.onnx',
                  example_outputs=output)

import torch
import torch.nn as nn

class test(nn.Module):
    def __init__(self, rnn_dims=32):
        super().__init__()
        self.rnn = nn.LSTM(input_size=rnn_dims, hidden_size=rnn_dims)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out


model = test()

scripted_model = torch.jit.script(model)

import torch
import torch.nn as nn

class testSubMod(nn.Module):
    def __init__(self, rnn_dims=32):
        super().__init__()
        self.lin = nn.Linear(rnn_dims, rnn_dims, bias=False)
        
    def forward(self, out, num_loops):
        for _ in torch.arange(num_loops):
            out = self.lin(out)
        return out
        

class test(nn.Module):
    def __init__(self, rnn_dims=32):
        super().__init__()
        self.submod = torch.jit.script(testSubMod())
        
    def forward(self, x):
        out = torch.ones([x.size(0), x.size(1)],
                         dtype=x.dtype, device=x.device)
        return self.submod(out, torch.tensor(10))

model = test()
model = torch.jit.script(model)

input = torch.ones((32, 32, 32)).float()
output = model(input)

torch.onnx.export(model, input,
                  'test.onnx', example_outputs=output)

import onnx

onnx_model = onnx.load("test.onnx")
print(onnx.helper.printable_graph(onnx_model.graph))

import torch
import torch.nn as nn

class testSubMod(nn.Module):
    def __init__(self, rnn_dims=32):
        super().__init__()
        self.rnn = nn.LSTM(input_size=rnn_dims, hidden_size=rnn_dims)

    def forward(self, out, num_loops):
        for _ in torch.arange(num_loops):
            out, _ = self.rnn(out)
        return out
    
class test(nn.Module):
    def __init__(self, rnn_dims=32):
        super().__init__()
        self.submod = testSubMod()

    def forward(self, x):
        return self.submod(x, torch.tensor(10))


model = test()

input = torch.ones((32, 32, 32)).float()
output = model(input)

torch.onnx.export(model, input,
                  'test.onnx', example_outputs=output)

import onnx

onnx_model = onnx.load("test.onnx")
print(onnx.helper.printable_graph(onnx_model.graph))