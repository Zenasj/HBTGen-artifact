import torch

# Script to create a reproducible accuracy issue with my model.
kwargs = {"fastmath_mode": True}
exp_program = export(my_model, sample_inputs, kwargs)
result = exp_program.module()(*sample_inputs, **kwargs)
# Uhoh, I dont like that result, lets send the module to a colleague to take a look.
torch.export.save(exp_program, "my_model.pt2")

# Script to load and reproduce results from a saved ExportedProgram.
loaded_program = torch.export.load("my_model.pt2")
# The following line is enabled by this Diff, we pull out the arguments
# and options that caused the issue.
args, kwargs = loaded_program.example_inputs
reproduced_result = loaded_program.module()(*args, **kwargs)
# Oh I see what happened here, lets fix it.