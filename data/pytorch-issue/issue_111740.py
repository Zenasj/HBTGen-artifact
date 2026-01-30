import torch
import torch.nn as nn

hidden_states = torch.randn([4, 2048, 512])
v_proj = nn.Linear(512, 128, bias=False)
value_states = v_proj(hidden_states)

h1, h2 = torch.chunk(hidden_states, 2, dim=0)
v1 = v_proj(h1)

assert h1.equal(hidden_states[:2])
print(v1[0,0,0].item())
print(value_states[0,0,0].item())
assert v1.equal(value_states[:2])

hidden_states = torch.randn([4, 2048, 512]).cuda()
v_proj = nn.Linear(512, 128, bias=False).cuda()
value_states = v_proj(hidden_states)

h1, h2 = torch.chunk(hidden_states, 2, dim=0)
v1 = v_proj(h1)

assert h1.equal(hidden_states[:2])
print(v1[0,0,0].item())
print(value_states[0,0,0].item())
assert v1.equal(value_states[:2])

import torch
import torch.nn as nn

hidden_states = torch.randn([4, 2048, 512])
v_proj = nn.Linear(512, 128, bias=False)
value_states = v_proj(hidden_states)

h1, h2 = torch.chunk(hidden_states, 2, dim=0)
v1 = v_proj(h1)

torch.testing.assert_close(h1, hidden_states[:2])
print(v1[0,0,0].item())
print(value_states[0,0,0].item())
torch.testing.assert_close(v1, value_states[:2])

hidden_states = torch.randn([4, 2048, 512]).cuda()
v_proj = nn.Linear(512, 128, bias=False).cuda()
value_states = v_proj(hidden_states)

h1, h2 = torch.chunk(hidden_states, 2, dim=0)
v1 = v_proj(h1)

torch.testing.assert_close(h1, hidden_states[:2])
print(v1[0,0,0].item())
print(value_states[0,0,0].item())
torch.testing.assert_close(v1, value_states[:2])