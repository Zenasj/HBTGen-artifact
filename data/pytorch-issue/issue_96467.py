import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels, num_masks=2):
        mask_positions = labels != -100

        # outputs = outputs[mask_positions]  # not allowed as dynamic shape op
        # torch.masked_select(labels, mask_positions)  # also not allowed as a dynamic shape operator

        indices = torch.argsort(mask_positions.int())[-num_masks:]  # ugh

        outputs = outputs[indices]  # not allowed as dynamic shape op, but ok with indices
        labels = labels[indices]

        return {"loss": outputs.mean(), "outputs": outputs}


labels = torch.ones((16,), dtype=torch.long) * -100
labels[3] = 8
labels[5] = 3
outputs = torch.randn(16, 128)
model = torch.compile(Model(), fullgraph=True, dynamic=True)
loss = model(outputs, labels)["loss"]