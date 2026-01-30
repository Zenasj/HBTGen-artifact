import torch

ep = torch.export.export(model, example_inputs)
package = torch._inductor.aoti_compile_and_package(ep, inductor_configs=inductor_configs)
compiled = torch._inductor.aoti_load_package(package)

print(compiled.get_constant_fqns())  # see what are the fqns needed/available

compiled.load_constants(new_state_dict, check_full_update=True)  # update the constants in AOTI