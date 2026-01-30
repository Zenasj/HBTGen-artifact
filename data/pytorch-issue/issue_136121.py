self.tok_embeddings = nn.Embedding(32, 768)
self.norm = nn.LayerNorm(768)
self.output = nn.Linear(768, 32, bias=False)

import torch
import torch.nn as nn
import copy

class EmbNormLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(32, 768)
        self.norm = nn.LayerNorm(768)
        self.output = nn.Linear(768, 32, bias=False)

    def forward(self, tokens):
        _, seq_len = tokens.size()
        h = self.tok_embeddings(tokens)
        h = self.norm(h)
        output = self.output(h).float()
        return output

backend = "inductor" # "aot_eager" works fine
torch.use_deterministic_algorithms(True)

for i in range(100):
    torch.manual_seed(42)
    eager_model = EmbNormLinear().cuda()
    compiled_model = torch.compile(copy.deepcopy(eager_model), backend=backend)

    inp = torch.randint(0, 32, (16, 16), device="cuda")

    eager_model(inp).sum().backward()
    eager_param_sum = torch.stack([p.sum() for p in eager_model.parameters()]).sum() 
    eager_grad_sum = torch.stack([p.grad.sum() for p in eager_model.parameters()]).sum()

    compiled_model(inp).sum().backward()
    compiled_param_sum = torch.stack([p.sum() for p in compiled_model.parameters()]).sum() 
    compiled_grad_sum = torch.stack([p.grad.sum() for p in compiled_model.parameters()]).sum()

    assert torch.equal(eager_param_sum, compiled_param_sum), f"{i=} {eager_param_sum} vs {compiled_param_sum}"
    assert torch.equal(eager_grad_sum, compiled_grad_sum), f"{i=} {eager_grad_sum} vs {compiled_grad_sum}"
    # print(f"{compiled_grad_sum=} {eager_grad_sum=}")