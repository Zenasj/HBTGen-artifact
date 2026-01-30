import torch as to
import torch.nn as nn


class Mod(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x): 
        return self.fc(x)


def main():
    model = Mod()
    x = to.rand(1, 3)
    traced_module = to.jit.trace(model, x)

    traced_module.save('traced_model.pt')

main()

#${TORCHPATH}/lib/libtorch.a ${TORCHPATH}/lib/libc10.a ${TORCHPATH}/lib/libnnpack.a ${TORCHPATH}/lib/libcpuinfo.a
#${TORCHPATH}/lib/libeigen_blas.a ${TORCHPATH}/lib/libpytorch_qnnpack.a