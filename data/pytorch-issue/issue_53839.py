import torch
import torch.nn as nn

class MyModuleFork(nn.Module):
    def forward(self) -> futures.Future[torch.Tensor]:
        fut = torch.jit.fork(torch.rand, 2, 2)
        return fut

class MyModuleFut(nn.Module):
    def forward(self) -> futures.Future[torch.Tensor]:
        fut = futures.Future() # jit compile error 
        fut.set_result(torch.rand(2, 2))
        return fut

torch.jit.script(MyModuleFork()) # works fine 
torch.jit.script(MyModuleFut()) # error