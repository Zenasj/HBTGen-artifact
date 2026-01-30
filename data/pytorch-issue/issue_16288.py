import torch

class ListOfTupleOfTensor(torch.jit.ScriptModule):
    def __init__(self):
        super(ListOfTupleOfTensor, self).__init__()

    @torch.jit.script_method
    def forward(self, x):
        # type: (Tensor) -> List[Tuple[Tensor, Tensor]]

        returns = torch.jit.annotate(List[Tuple[Tensor, Tensor]], [])
        for i in range(10):
            returns.append((x, x))

        return returns

c = ListOfTupleOfTensor()
print(c.graph)