import torch

traced_script_module = torch.jit.trace(model.module, (imgL_prep, imgR_prep), check_trace = False)
traced_script_module.save("stereonet_traced.pt")