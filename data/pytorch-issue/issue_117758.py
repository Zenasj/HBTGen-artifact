from functools import partial

import torch
import torch.nn as nn

device = torch.device("cuda", 0)
x = torch.randn(1024, 64, device=device)

for trigger_compilation in [False, True]:
    print("trigger_compilation:", trigger_compilation)

    # Setup
    model = nn.Sequential(
        nn.Linear(64, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 64),
        nn.ReLU(),
    )
    model.train()
    model.to(device)
    model = torch.compile(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # Trigger compilation
    if trigger_compilation:
        model(x)

    # Register hook
    @torch._dynamo.disable
    def hook(m, i, o, *, idx, dest):
        dest[idx] = o.sum()
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    handles = []
    dest = {}
    for idx, linear in enumerate(linears):
        h = linear.register_forward_hook(partial(hook, idx=idx, dest=dest))
        handles.append(h)

    # Run
    for _ in range(4):
        dest.clear()
        y = model(x)
        loss = torch.mean((x - y) ** 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss.item(), {k: v.item() for k, v in dest.items()})

    # Cleanup
    for h in handles:
        h.remove()
    
    print()