import torch

class M(torch.jit.ScriptModule):
    def __init__(self):
        super(M, self).__init__()

    @torch.jit.script_method
    def forward(self, token: str) -> List[str]:
        return list(token)

b = M()