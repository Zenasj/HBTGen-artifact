import torch

# user_model is in cuda
dynamicLib_path = torch._export.aot_compile(
    user_model,
    args = tuple(inputs_list),
    dynamic_shapes = {**dynamic_shapes},
    options={
            "aot_inductor.output_path": os.path.join(args.dynamicLib_folder, dynamicLib_name), 
            "max_autotune": True
            },
)