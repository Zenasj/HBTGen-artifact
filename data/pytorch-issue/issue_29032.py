import torch
import torch.nn as nn


class modelA(nn.Module):
    def __init__(self):
        super(modelA, self).__init__()

        self.observer = torch.quantization.MinMaxObserver()

    def forward(self, input):
        self.observer(input)

class modelB(modelA):
    def __init__(self):
        super(modelB, self).__init__()

        self.observer_ = torch.quantization.MinMaxObserver()


if __name__ == '__main__':

    # model A
    a = modelA()

    # do something
    input = torch.randn(32)
    a(input)

    # save a
    torch.save(a.state_dict(), "a.pt")

    # define model B
    b = modelB()
    target_model_dict = b.state_dict()

    # Now we want to initilize all params in B with those from A.
    # Here I follow the steps as in https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3

    # 0. loading pre-trained dict
    source_dict = torch.load("a.pt")

    # 1. filter out unnecessary keys
    source_dict = {k: v for k, v in source_dict.items() if k in target_model_dict}

    # 2. overwrite entries in the existing state dict
    target_model_dict.update(source_dict) 

    # 3. load the new state dict
    not_found = b.load_state_dict(source_dict, strict=False) # <-- will crash

    print(not_found)