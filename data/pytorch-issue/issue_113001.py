import torch

def trace_handler(p):
      p.export_chrome_trace(os.path.join(args.perf_torch_dir, 'trace.json'))

profiler = torch.profiler.profile(
      activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
      schedule=torch.profiler.schedule(skip_first=0, wait=0, warmup=1, active=2, repeat=0),
      on_trace_ready=trace_handler,
      record_shapes=True,
      profile_memory=True,
      with_stack=True,
      with_modules=True,
      use_cuda=True
)
profiler.start()
profiler.step()
profiler.stop()