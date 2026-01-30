import torch._inductor.config as inductor_config
inductor_config.profiler_mark_wrapper_call = True
inductor_config.cpp.enable_kernel_profile = True

inductor_config.cpp.descriptive_names = "inductor_node"