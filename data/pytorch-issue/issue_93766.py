import torch

def save_fx(gm, example_inputs):
    from functorch.compile import aot_module, aot_module_simplified

    def graph_saver_forward(gm, _):
        gm.to_folder(...)
        return gm

    def graph_saver_backward(gm, _):
        gm.to_folder(...)
        return gm

    return aot_module(gm, fw_compiler=graph_saver_forward, bw_compiler=graph_saver_backward) 

optimize_ctx = torchdynamo.optimize(    
            save_fx, # aot_module(gm, fw_compiler=graph_saver_forward, bw_compiler=graph_saver_backward)
            nopython=args.nopython,
        )

with optimize_ctx:
           model_iter_fn(model, example_inputs, collect_outputs=False)