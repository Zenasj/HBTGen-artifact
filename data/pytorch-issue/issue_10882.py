import torch
import torch.nn as nn

class NEModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100,10)

    def forward(self, x):
        emb = self.embedding(x)
        mask = torch.ne(x, 0)
        return emb * mask.unsqueeze(2).expand_as(emb).float()


x = torch.randint(0,100,(32,10),dtype=torch.long)
mod = NEModule()
torch.onnx.export(mod, x, "ne.onnx", verbose=True)

def expand(g, self, size, implicit):
    if _is_value(size):
        shape = size
    else:
        if self.isTensor():
            self_sizes = self.type().sizes()
            if self_sizes and len(size) == 2 and self_sizes[0] == size[0]:
                return g.op("Flatten", self, axis_i=1)
        shape = g.op("Constant", value_t=torch.LongTensor(size))
    return g.op("Expand", self, shape)