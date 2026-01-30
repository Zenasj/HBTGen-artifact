import torch

with tprofiler.profile(model, use_cuda=True, use_kineto=True) as prof:
        with tprofiler.record_function('Overall'):
            output = model(input_batch)
            torch.cuda.synchronize()