import torch.nn as nn

import torch


class Model(torch.nn.Module):
    def forward(self, x):
        x = x.clone()
        x.index_put_(
            indices=(torch.LongTensor([0, -1]), torch.LongTensor([-2, 1])),
            values=torch.Tensor([1.0, 5.0]),
        )
        return x
    

if __name__ == "__main__":
    SHAPE = (3, 4)

    model = Model()
    model.eval()

    x = torch.rand(SHAPE)
    exported_program = torch.export.export(model, (x,))
    graph_module = exported_program.module()

    x = torch.rand(SHAPE)
    y = graph_module(x)
    print(y)

FakeTensor(..., size=(3, 4))

tensor([[0.8517, 0.5608, 1.0000, 0.3297],
        [0.1325, 0.2706, 0.1296, 0.6235],
        [0.2908, 5.0000, 0.3512, 0.2101]])

tensor([[0.2307, 0.5860, 1.0000, 0.8029],
        [0.4501, 0.3431, 0.3892, 0.0574],
        [0.9867, 5.0000, 0.0976, 0.8971]])