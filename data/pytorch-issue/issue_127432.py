import torch

# export, fail, parse & refine suggested fixes, re-export
try:
    export(model, inps, dynamic_shapes=dynamic_shapes)
except torch._dynamo.exc.UserError as exc:
    new_shapes = parse_and_refine_suggested_fixes(exc.msg, dynamic_shapes)
    export(model, inps, dynamic_shapes=new_shapes)