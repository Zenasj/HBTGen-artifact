import torch

# Assuming we have a model of the name 'model'

example_input = torch.rand(1, 3, 224, 224)

# enable oneDNN Graph
torch.jit.enable_onednn_fusion(True)
# Disable AMP for JIT
torch._C._jit_set_autocast_mode(False)
with torch.no_grad(), torch.cpu.amp.autocast():
    model = torch.jit.trace(model, (example_input))
    model = torch.jit.freeze(model)
     # 2 warm-ups (2 for tracing/scripting with an example, 3 without an example)
    model(example_input)
    model(example_input)
    
    # speedup would be observed in subsequent runs.
    model(example_input)