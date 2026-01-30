import torch

records = torch.autograd._disable_profiler()
events_average = EventList(parse_cpu_trace(records)).total_average()
cpu_time = events_average.cpu_time_total / 1000
cuda_time = events_average.cuda_time_total / 1000