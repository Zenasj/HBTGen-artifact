import torch.nn as nn

import torch

class TestModule(torch.jit.ScriptModule):
    def __init__(self):
        super(TestModule, self).__init__()
        self.d = torch.nn.Dropout()

    @torch.jit.script_method
    def forward(self, input):
        return self.d(input)

a = TestModule()
a.eval()
torch.jit.save(a, "test_training_flag.pt")

b = torch.jit.load("test_training_flag.pt")
print("a's Training: {}\nb's Training: {}".format(a.training, b.training))
print("a.d's Training: {}\nb.d's Training: {}".format(a.d.training, b.d.training))

input = torch.zeros([10])
print("a(input): ", a(input))
print("b(input): ", b(input))