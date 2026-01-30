import torch.nn as nn

import torch._inductor.config

torch._inductor.config.coordinate_descent_tuning = True

compiled_model = get_model(config, "cuda")
compiled_model.to(torch.bfloat16)
compiled_model.decoder.model.forward = torch.compile(compiled_model.decoder.model.forward, mode="reduce-overhead", fullgraph=True)  # type: ignore


# %%
from torch.nn.attention import SDPBackend

img = torch.randn(1, 1, 1536, 1152).to("cuda")
with torch.inference_mode():
    # with torch.autocast("cuda", dtype=torch.bfloat16):
    with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]):
        timer = Timer(name="Inference with compile (warmup)")
        compiled_model.generate(img.to(torch.bfloat16), timer=timer.gen_child("Generate"), max_length=10)
        compiled_model.generate(img.to(torch.bfloat16), timer=timer.gen_child("Generate"), max_length=10)
        compiled_model.generate(img.to(torch.bfloat16), timer=timer.gen_child("Generate"), max_length=10)
        compiled_model.generate(img.to(torch.bfloat16), timer=timer.gen_child("Generate"), max_length=10)
        print(output.shape)