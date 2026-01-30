import torch

def custom_aot_backend(gm, example_inputs):
    functorch.compile.config.use_functionalize = True
    functorch.compile.config.use_fake_tensor = use_fake
    return aot_autograd(
        fw_compiler=custom_compiler_inner,
        bw_compiler=custom_compiler_inner,
    )(gm, example_inputs)

model_compiled = torch.compile(model, backend=custom_aot_backend)

def iteration(x, y):
    optimizer.zero_grad()
    result = model_compiled(x)
    loss = result.sum().abs()
    loss.backward()
    optimizer.step()
    return loss, result