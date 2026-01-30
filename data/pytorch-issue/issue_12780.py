import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn
from torch.functional import F

class ACTNN(torch.jit.ScriptModule):

    __constants__ = ['hidden_depth', 'hidden_width']
    # hidden_width = 125
   # hidden_depth = 3
    # output_size = 80
    # input_size = 125
    def __init__(self, input_size, hidden_width, output_size, hidden_depth):
        super(ACTNN, self).__init__()
        self.hidden_depth = hidden_depth
        self.hidden_width = hidden_width
        self.rnn = torch.jit.trace(nn.RNN(input_size, hidden_width,
                                          num_layers=self.hidden_depth, batch_first=True), example_inputs=(torch.rand(1, 1, input_size), torch.rand(self.hidden_depth, 1, hidden_width)))

        self.proc = torch.jit.trace(nn.Linear(hidden_width, output_size), example_inputs=torch.rand(1, 1, hidden_width))


    @torch.jit.script_method
    def forward(self, input: torch.Tensor, hidden: torch.Tensor):

        input = input.view(1, 1, -1)
        hidden = hidden.view(self.hidden_depth, 1, self.hidden_width)
        # Define the list objects
        states = []
        hiddens = []
        halt_probs = []
        n = 0
        rnn_out, hidden = self.rnn(input, hidden)
        state = self.proc(rnn_out)
        hiddens.append(hidden)
        states.append(state)
        halt_probability = F.sigmoid(hiddens[n].sum())
        halt_probs.append(halt_probability)
       # At this point, both the states & halt_probs lists will contain:
       # 1x 1,1,80 wide tensor
       # 1x 3,1,125 wide tensor
       # 1x 1 wide tensor

        print(states)
        print(halt_probs)
        residual = torch.sum(torch.cat(halt_probs))

        states_tensor = torch.stack(states, dim=1)
        hiddens_tensor = torch.stack(hiddens, dim=1)
        halt_subset = halt_probs[0:n-1]
        halt_subset.append(residual)
        halt_prob_tensor = torch.stack(halt_subset).view(-1)
        output = torch.mv(states_tensor, halt_prob_tensor)
        hidden = torch.mv(hiddens_tensor, halt_prob_tensor)
        return output, hidden

import torch

from torch.jit import script


@script
def update_tensor(tensor):
    tensor_prime = []
    for i in range(len(tensor)):
        n = tensor[i]
        tensor_prime.append(n)
    nodes_prime = torch.stack(tensor_prime)
    return nodes_prime


def main():
    tensor = torch.FloatTensor(4, 2).random_()

    for i in range(3):
        print("="*20)
        print(tensor.shape)
        print(tensor)
        tensor = update_tensor(tensor)


if __name__ == "__main__":
    main()

import torch
from torch import nn

from torch.jit import ScriptModule, script_method


class ItsAList(ScriptModule):

    def __init__(self):
        super(ItsAList, self).__init__()

    @script_method
    def forward(self, tensor):
        just_a_list = []
        return just_a_list


class RebuildSeveralTimes(ScriptModule):
    __constants__ = ['module_list', 'times']

    def __init__(self, times):
        super(RebuildSeveralTimes, self).__init__()
        self.times = times
        alist_modules = []
        for i in range(times):
            alist_modules.append(ItsAList())
        self.module_list = nn.ModuleList(alist_modules)

    @script_method
    def forward(self, tensor):
        for _ in self.module_list:  # works as expected with: for _ in range(self.times):
            tensor_list = []
            for i in range(tensor.size(0)):
                n = tensor[i]
                tensor_list.append(n)
            tensor = torch.stack(tensor_list)
        return tensor


for i in range(4):
    tensor = torch.FloatTensor([i])
    model = RebuildSeveralTimes(i)
    new_tensor = model(tensor)
    print(f"i {i}", "="*20)
    print(tensor)
    print(new_tensor)