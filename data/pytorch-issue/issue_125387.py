import torch
import torch._dynamo as dynamo
import torch._inductor as inductor
import torch.nn as nn

dynamo.reset()


# RETURN_INDEX = True: the outputs are 0, 1, 2, 3, 4, 5 (this is expected)
# RETURN_INDEX = False: the outputs are 2, 3, 4, 5, 5, 5 (it should be 2, 3, 4, 5, 6, 7)
RETURN_INDEX = True

class ToyModel(torch.nn.Module):
    def __init__(self, return_index):
        super(ToyModel, self).__init__()
        self.value = -1
        self.return_index = return_index
        self.cache = torch.tensor([2, 3, 4, 5, 6, 7])

    def forward(self, value):
        self.value += 1
        if self.return_index:
            return self.value  # the outputs are: 0, 1, 2, 3, 4, 5
        else:
            return self.cache[self.value]  # the outputs are:  2, 3, 4, 5, 5, 5

model = ToyModel(return_index=RETURN_INDEX )
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

values = [6, 8, 10, 12, 13, 14]
for value in values:
    output = model.forward(value)
    print(f"output = {output}")

RETURN_INDEX = True
model = ToyModel(return_index=RETURN_INDEX )
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

values = [6, 8, 10, 12, 13, 14]
for value in values:
    output = model.forward(value)
    print(f"output = {output}")



RETURN_INDEX = False
model = ToyModel(return_index=RETURN_INDEX )
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

values = [6, 8, 10, 12, 13, 14]
for value in values:
    output = model.forward(value)
    print(f"output = {output}")