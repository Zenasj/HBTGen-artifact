import torch.nn as nn

import torch

class MyCell(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

def main():
    my_cell = MyCell()
    my_cell.cuda().to(memory_format=torch.contiguous_format)
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(my_cell.parameters(), lr=0.01)
    x = torch.rand(3, 4, device='cuda')
    h = torch.rand(3, 4, device='cuda')
    target = torch.rand(3, 4, device='cuda')
    logits, _ = my_cell(x, h)
    loss = loss_fn(logits, target)
    loss.backward(retain_graph=True)
    optimizer.step()

    ## Capture a trace of the network
    traced_cell = torch.jit.trace(my_cell, (x,h))
    traced_loss = torch.jit.trace(loss_fn, (logits, target))
    print("Printing Graph view of  model")
    print(traced_cell.graph)
    print("Printing Inlinded Graph view of  model")
    print(traced_cell.inlined_graph)
    print("Printing Inlinded Graph view of  loss")
    print(traced_loss.inlined_graph)
    print ("Printing Params -> {}".format(my_cell.named_parameters()))
    for param in my_cell.named_parameters():
        print("Param -> {}".format(param))

    graph_nodes = traced_cell.graph.nodes()
    ## Printing each node in graph works
    print("Printing each Node in graph works")
    for node in graph_nodes:
        print ("Node -> {}".format(node))

    inlined_nodes = traced_cell.inlined_graph.nodes()
    ## Printing each node in graph works
    print("Printing each Node in inlinde graph - fails w/ segfault")
    for node in inlined_nodes:
        print ("Node -> {}".format(node))


if __name__ == "__main__":
    main()