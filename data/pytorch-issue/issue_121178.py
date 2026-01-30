import torch
torch._dynamo.config.accumulated_cache_size_limit = 1
torch._dynamo.config.cache_size_limit = 1

def training_loop():
    for _ in range(2):
        params = [torch.rand(4, 5, device="cuda") for _ in range(3)]
        kwargs = {"foreach": True, "capturable": False}
        optimizer = torch.optim.Adamax(params, **kwargs)

        optimizer.zero_grad()
        for p in params:
            p.grad = torch.ones_like(p)

        optimizer.step()

compiled_training_loop = torch._dynamo.optimize("eager")(training_loop)
compiled_training_loop()