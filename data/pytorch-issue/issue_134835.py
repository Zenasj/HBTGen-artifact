import torch

[tasklist]
### Tasks

exported_gm = torch._export.aot_compile(
                    gm,
                    args = tuple([data]),
                    # dynamic_shapes = dynamic_shapes,
                    options={
                            "aot_inductor.output_path": 'export_gpu.so',
                            "triton.cudagraphs": False,
                            "max_autotune": True
                    },
        )