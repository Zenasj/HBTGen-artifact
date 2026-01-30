import threading

import torch

class MyModule(torch.jit.ScriptModule):
    """Instantiating this outside the main thread causes a deadlock."""

    def __init__(self):
        super(MyModule, self).__init__()

    @torch.jit.script_method
    def forward(self, action: torch.Tensor):
        return torch.ones((100, 100)) + action

def run():
    print('Creating ScriptModule ...')
    module = MyModule()
    print('done. Without deadlock.')
    module(torch.tensor(0))
    print('Ran ScriptModule')

thread = threading.Thread(target=run)

thread.start()

thread.join()