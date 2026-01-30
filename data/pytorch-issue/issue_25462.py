import torch

@torch.jit.script
class A:
    def __init__(self):
        self.v = 100
    
    def copy(self, b: 'A'):
        self.v = b.v