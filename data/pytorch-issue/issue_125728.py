import torch._inductor.config
# torch._inductor.config.trace.enabled = True
torch._inductor.config.max_autotune_gemm_backends = "TRITON"
torch._inductor.config.max_autotune = True
torch._inductor.config.max_autotune_conv_backends = "TRITON" # <- this makes it fail.