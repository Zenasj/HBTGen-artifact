import torch.nn as nn

3
import torch
import pickle
import copy
import torch, torch.fx.experimental.meta_tracer

class MetaTracerTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings=42, embedding_dim=16)
        self.layernorm = torch.nn.LayerNorm(16)

    def forward(self, x):
        emb = self.emb(x)
        emb = emb + torch.arange(emb.shape[-1], dtype=torch.float, device=emb.device)
        lol = self.layernorm(emb)
        return torch.relu(lol) if lol.shape[0] < 30 else torch.sigmoid(lol)

mttm = MetaTracerTestModule()
x = torch.zeros(16, dtype=torch.long).random_(42)
gm = torch.fx.experimental.meta_tracer.symbolic_trace(mttm, meta_args={'x' : x.to(device='meta')})
# gm._tracer_extras => {'meta_args': {'x': tensor(..., device='meta', size=(16,), dtype=torch.int64)}}

# Test serialization/deserialization of copy
gm_copy = copy.deepcopy(gm)
# gm_copy._tracer_extras => {}

with open('test.pkl', 'wb') as f:
    pickle.dump(gm_copy, f)

with open('test.pkl', 'rb') as f:
    loaded_copy = pickle.load(f) # ERROR: MetaTracer.trace() missing 1 required positional argument: 'meta_args'