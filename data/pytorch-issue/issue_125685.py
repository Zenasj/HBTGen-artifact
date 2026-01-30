import torch

with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        on_trace_ready=trace_handler,
        experimental_config=torch.profiler._ExperimentalConfig(
            profiler_metrics=[
                "kineto__tensor_core_insts",
                "dram__bytes_read.sum",
                "dram__bytes_write.sum"],
            profiler_measure_per_kernel=False),
    ) as prof:
        res = train_batch(modeldef)
        prof.step()