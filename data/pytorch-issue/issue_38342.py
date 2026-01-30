import torch
import torch.nn as nn
import time


class custom_rnn(torch.jit.ScriptModule):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
        nonlinearity="relu",
        device="cuda",
    ):

        super(custom_rnn, self).__init__()
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.device = device

        self.w = nn.Linear(
            self.input_size, 2 * self.hidden_size, bias=False
        ).to(device)
        
        # Initilizing initial state h
        self.h_init = torch.zeros(
                self.batch_size,
                self.hidden_size,
                requires_grad=False,
                device=self.device,
            )

    @torch.jit.script_method
    def forward(self, x):
        ht = self.h_init
        
        # Loop over time axis
        for k in range(x.shape[1]):
            ht = ht + 1.0
        return ht

def run():

    inp_tensor = torch.rand([4, 500, 40]).to('cuda')
    net = custom_rnn(40, 512, 1, 4, device='cuda').to('cuda')
    start = time.time()

    for i in range(1000):
        out_tensor = net(inp_tensor)

    end = time.time()
    print(end - start)


print("DEFAULT")
run()

torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)

print("PE")
run()

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

print("LE")
run()

torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(False)

print("SE")
run()

print("Version: ", torch.__version__)

import torch
import torch.nn as nn
import time


class custom_rnn(torch.jit.ScriptModule):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
        nonlinearity="relu",
        device="cuda",
    ):

        super(custom_rnn, self).__init__()
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.device = device

        self.w = nn.Linear(
            self.input_size, 2 * self.hidden_size, bias=False
        ).to(device)
        
        # Initilizing initial state h
        self.h_init = torch.zeros(
                self.batch_size,
                self.hidden_size,
                requires_grad=False,
                device=self.device,
            )

    @torch.jit.script_method
    def forward(self, x):
        ht = self.h_init
        
        # Loop over time axis
        for k in range(x.shape[1]):
            ht = ht + 1.0
        return ht

def get_fresh():
    return torch.jit.load('scriptmodule.pt')


def run(net, inp_tensor):
    

    start = time.time()

    
    for i in range(1000):
        out_tensor = net(inp_tensor)

    end = time.time()
    print(end - start)


def warmup(net, inp_tensor):
    for i in range(20):
        out_tensor = net(inp_tensor)

net = custom_rnn(40, 512, 1, 4, device='cuda').to('cuda')
torch.jit.save(net, 'scriptmodule.pt')
inp_tensor = torch.rand([4, 500, 40]).to('cuda')

print("DEFAULT")
net = get_fresh()
warmup(net, inp_tensor)
run(net, inp_tensor)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)

print("PE")
net = get_fresh();
warmup(net, inp_tensor)
run(net, inp_tensor)

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

print("LE")
net = get_fresh();
warmup(net, inp_tensor)
run(net, inp_tensor)

torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(False)

print("SE")
net = get_fresh();
warmup(net, inp_tensor)
run(net, inp_tensor)

print("Version: ", torch.__version__)