import time
import torch
import torch.nn as nn
from optree import tree_map

torch.manual_seed(0)

# torch.set_default_dtype(torch.double)

def test_func():
    def make_mlp(**kwargs):
        layers = []
        last_dim = kwargs["in_features"]
        for i in range(len(kwargs["num_cells"])):
            layers.append(nn.Linear(last_dim, kwargs["num_cells"][i]))
            layers.append(kwargs["activation_class"]())
            last_dim = kwargs["num_cells"][i]
        layers.append(nn.Linear(last_dim, kwargs["out_features"]))
        return nn.Sequential(*layers)

    mlp_kwargs = {
        "num_cells": [256, 256, 256],
        "in_features": 30,
        "out_features": 1,
        "activation_class": nn.ReLU,
    }

    critic_1 = make_mlp(**mlp_kwargs)
    critic_2 = make_mlp(**mlp_kwargs)

    # compile vmap function
    critic_params = tree_map(lambda *x: torch.stack(x), *[critic_1.state_dict(), critic_2.state_dict()])
    critic_call = lambda params, inputs: torch.func.functional_call(critic_1, params, inputs)
    critic_call_vmap = lambda x: torch.vmap(critic_call, (0, None), randomness="same")(critic_params, x)
    critic_call_vmap_compile = torch.compile(critic_call_vmap)

    # compile separate call
    critic_call_separate = lambda x: torch.stack([critic_1(x), critic_2(x)])
    critic_call_separate_compile = torch.compile(critic_call_separate)

    # generate random inputs
    batch_size = 4096
    x = torch.randn(batch_size, mlp_kwargs["in_features"])

    # call once to compile
    for i in range(1):
        y_separate = critic_call_separate_compile(x)
        y_vmap = critic_call_vmap_compile(x)

    # separate forward
    start = time.time()
    y_separate = critic_call_separate(x)
    t_separate = time.time() - start

    # separate forward with compile
    _ = critic_call_separate_compile(x)
    _ = critic_call_separate_compile(x)

    start = time.time()
    y_separate_compile = critic_call_separate_compile(x)
    t_separate_compile = time.time() - start

    # vmap forward
    _ = critic_call_vmap(x)
    _ = critic_call_vmap(x)

    start = time.time()
    y_vmap = critic_call_vmap(x)
    t_vmap = time.time() - start

    # vmap forward with compile
    _ = critic_call_vmap_compile(x)
    _ = critic_call_vmap_compile(x)

    start = time.time()
    y_vmap_compile = critic_call_vmap_compile(x)
    t_vmap_compile = time.time() - start

    print("time separate", t_separate)
    print("time separate compile", t_separate_compile)
    print("time vmap", t_vmap)
    print("time vmap compile", t_vmap_compile)
    print()
    print("y_separate == y_separate_compile:", torch.all(y_separate == y_separate_compile))
    print("y_vmap == y_vmap_compile:", (y_vmap - y_vmap_compile).norm())
    print("y_separate == y_vmap:", (y_separate - y_vmap).norm())
    print()
    print("y_separate, y_vmap are close:", torch.allclose(y_separate, y_vmap, atol=1e-6))
    print("y_vmap, y_vmap_compile are close:", torch.allclose(y_vmap, y_vmap_compile, atol=1e-6))


test_func()