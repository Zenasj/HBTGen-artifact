import torch.nn as nn

import torch 

class SampleModel(torch.nn.Module):
    def __init__(self):
        super(SampleModel).__init__()
        self.__attribute = None
    
    def forward(x):
        return x

class SampleManager:
    def __init__(self):
        self.model = SampleModel()

    def crash(self):
        print(self.model.__attribute)


if __name__ == "__main__":
    m = SampleManager()
    m.crash()