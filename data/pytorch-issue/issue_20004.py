import torch

@torch.jit.script
class Pair:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def sum(self):
        return self.first + self.second

    def __getitem__(self, k):
        return self.first