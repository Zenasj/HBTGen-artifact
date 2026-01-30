import torch

class MyMod2(torch.jit.ScriptModule):
    __constants__ = ['mean', 'std']
    def __init__(self):
        super(MyMod2, self).__init__()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    @torch.jit.script_method
    def forward(self, input):
        mean = torch.tensor(self.mean)
        std = torch.tensor(self.std)
        return input.sub(mean[:, None, None]).div_(std[:, None, None])