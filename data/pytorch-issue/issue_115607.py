import torch.nn as nn

import torch
def training_loop():
    input = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ).reshape(3, 2)

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.Sigmoid(),
        torch.nn.Linear(3, 1),
        torch.nn.Sigmoid(),
    )

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params)

    for i in range(6):
        optimizer.zero_grad()
        # Test that step behaves as expected (a no-op) when grads are set to None
        if i != 3:
            output = model(input)
            loss = output.sum()
            loss.backward()

        optimizer.step()
        print("step", optimizer.state[params[0]]["step"])

compiled_training_loop = torch._dynamo.optimize("eager", save_config=False)(training_loop)

print("expected in eager:")
training_loop()

print("what actually happens after dynamo:")
compiled_training_loop()

import torch
input = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], device="cuda"
    ).reshape(3, 2)
params = [input]
input.grad = torch.ones_like(input)

def reset():
    global params, input
    input = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], device="cuda"
    ).reshape(3, 2)
    params = [input]
    input.grad = torch.ones_like(input)


def training_loop():
    optimizer = torch.optim.ASGD(params, foreach=False)

    for i in range(6):
        optimizer.zero_grad()
        #  Test that step behaves as expected (a no-op) when grads are set to None
        if i != 3:
            input.grad = torch.ones_like(input)

        optimizer.step()

compiled_training_loop = torch._dynamo.optimize("eager")(training_loop)

print("expected in EAGER:")
print(f"BEFORE the loop {params=}")
training_loop()
print(f"AFTER the loop {params=}")
reset()
print("=======================================")

print("what actually happens after DYNAMO:")
print(f"BEFORE the loop {params=}")
compiled_training_loop()
print(f"AFTER the loop {params=}")