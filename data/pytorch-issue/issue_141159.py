import torch
import torch.nn as nn

class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.true_graph_0 = GraphModule()



    def forward(self):
        true_graph_0 = self.true_graph_0
        return (true_graph_0,)