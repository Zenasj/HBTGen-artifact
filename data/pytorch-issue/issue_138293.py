import torch
import torch.nn as nn

class SimpleCopy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dst, src):
        dst.copy_(src)
        return dst

    def get_example_inputs(self):
        return (torch.randn(5, 5), torch.randn(5, 5))

mod = SimpleCopy()
with torch.no_grad():
    exported_program = export(mod.eval(), mod.get_example_inputs())
print (exported_program) # before
exported_program = exported_program.run_decompositions()
print (exported_program) # after