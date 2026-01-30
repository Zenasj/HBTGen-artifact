import torch.nn as nn

from torch.fx import GraphModule, Tracer
import torch
class GateEmbedding(torch.nn.Module):
    def __init__(self, in_size, out_size, act1=torch.nn.ReLU(), act2=torch.nn.Sigmoid(), act3=torch.nn.ReLU()):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_size, out_size)
        self.out_size = out_size
        self.act2 = act2
        print(f"debug ###{in_size}   {self.out_size}###")

    def forward(self, input):
        gate_out = self.act2(self.layer1(input))
        print(f"debug ###{gate_out.shape}   {self.out_size}###")
        size = gate_out.shape[0]
        print(f"debug ###{gate_out.shape[0]} {type(size)}  {self.out_size}###")
        gate_out_r = torch.reshape(gate_out, (gate_out.shape[0], self.out_size, -1))
        input_r = torch.reshape(input, (input.shape[0], self.out_size, -1))
        gate_out = (gate_out_r * input_r).reshape(input.shape[0], -1)
        # return self.act3(info_out * gate_out)
        return gate_out  # gate_2

my_tracer=Tracer()
print(f"torch version {torch.__version__}")
embedding_model=GateEmbedding(100, 41)
symbolic_traced : torch.fx.Graph = my_tracer.trace(embedding_model)
symbolic_traced.print_tabular()

from torch.fx import GraphModule, Tracer
import torch
class GateEmbedding(torch.nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 emb_size,
                 act1=torch.nn.ReLU(),
                 act2=torch.nn.Sigmoid(),
                 act3=torch.nn.ReLU()):
        super().__init__()
        self.layer1 = torch.nn.Linear(in_size, out_size)
        self.out_size = out_size
        self.emb_size = emb_size
        self.act2 = act2
        print(f"debug ###{in_size}   {self.out_size}###")

    def forward(self, input):
        gate_out = self.act2(self.layer1(input))
        print(f"debug ###{gate_out.shape}   {self.out_size}###")
        size = gate_out.shape[0]
        print(f"debug ###{gate_out.shape[0]} {type(size)}  {self.out_size}###")
        gate_out_r = torch.reshape(gate_out,
                                   (-1, self.out_size, 1))
        input_r = torch.reshape(input,
                               (-1, self.out_size, self.emb_size))
        gate_out = (gate_out_r * input_r).reshape(-1, self.out_size* self.emb_size)
        # return self.act3(info_out * gate_out)
        return gate_out  # gate_2


my_tracer = Tracer()
print(f"torch version {torch.__version__}")
embedding_model = GateEmbedding(100, 41, 16)
symbolic_traced: torch.fx.Graph = my_tracer.trace(embedding_model)
symbolic_traced.print_tabular()