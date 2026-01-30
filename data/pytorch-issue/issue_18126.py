import torch.nn as nn

import torch
import string

debug = True
device = 'cpu' if not torch.cuda.is_available() else 'cuda'
traced_generator = torch.jit.load('CharRNN_model.pt', device)


@torch.jit.script
def post_processing(output):
    output_dist = output.squeeze().div(0.8).exp()
    prob = torch.multinomial(output_dist, 1)[0]
    return prob


@torch.jit.script
def seq_loop(inputs, hidden, prediction_seq_length: int):
    out = []
    i = 0
    while i < prediction_seq_length:
        output, hidden = traced_generator(inputs, hidden)
        inputs = post_processing(output)
        out.append(inputs)
        i += 1
    return out

torch.jit.save(seq_loop, 'pytorch_issue.pt')

import torch
import string

debug = True
device = 'cpu' if not torch.cuda.is_available() else 'cuda'
traced_generator = torch.jit.load('CharRNN_model.pt', device)

def str2int(string_data):
    return [all_characters.index(c) for c in string_data]

def int2str(int_data):
    return ''.join([all_characters[i] for i in int_data])

class ScriptModuleWrapper(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.traced_generator = traced_generator

    @torch.jit.script_method
    def forward(self, inputs, hidden):    
        out = []
        i = 0
        n_layers = 2
        hidden_size = 300
        prediction_seq_length = 200
        while i < prediction_seq_length:
            output, hidden = self.traced_generator(inputs, hidden)
            inputs = self.post_processing(output)
            out.append(inputs)
            i += 1
        return out

    @torch.jit.script_method
    def post_processing(self, output):
        output_dist = output.squeeze().div(0.8).exp()
        prob = torch.multinomial(output_dist, 1)[0]
        return prob

filename = 'CharRNN_pipeline.pt'
print('Saving pipeline to {}'.format(filename))
ScriptModuleWrapper().save(filename)

class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.encoder = nn.Linear(2, 2)

    def forward(self, x):
        return self.encoder(x)

p = torch.ones(2, 2)
traced = torch.jit.trace(M(), (p,))

@torch.jit.script
def seq_loop(x):
    return traced(x)

torch.jit.save(seq_loop, 'pytorch_issue.pt')