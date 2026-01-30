import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 8)

    def forward(self, input_ids):
        outputs = self.embedding(input_ids)
        loss = outputs.mean()
        return dict(loss=loss, outputs=outputs)  # warning
        # return {"loss": loss, "outputs": outputs} ok

model = torch.compile(Model())
model(torch.randint(10, (4, 16)))