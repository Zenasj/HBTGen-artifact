import torch

torch._dynamo.debug_utils.MINIFIER_SPAWNED = True
compiler_fn = BACKENDS["dynamo_accuracy_minifier_backend"]

dynamo_minifier_backend = functools.partial(
    compiler_fn,
    compiler_name="inductor",
)       
opt_mod = torch._dynamo.optimize(dynamo_minifier_backend)(mod)

with torch.cuda.amp.autocast(enabled=False):
    opt_mod(*args)