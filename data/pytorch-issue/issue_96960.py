import torch

model, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=param_groups, dist_init_required=False)
model = torch.compile(model)

try:
    obj_ref = weakref.ref(guarded_object)
except TypeError:
    obj_ref = lambda: guarded_object