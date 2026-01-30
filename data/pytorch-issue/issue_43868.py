import torch
with torch.autograd.profiler.profile() as prof:
    with torch.autograd.profiler.record_function("##forward##") as rf:
        t1, t2 = torch.ones(1), torch.ones(1)
        t = torch.add(t1, t2)
        t = torch.mul(t, t)
        t = t.relu()
        t = t.sigmoid()

function_events = prof.function_events
rf_event = [e for e in function_events if "##forward##" in e.name][0]
remaining = set(function_events)  - {rf_event}
remaining_time = sum(e.self_cpu_time_total for e in remaining)
rf_time = rf_event.self_cpu_time_total

if rf_time < remaining_time:
    print(f"Record function scope time was {rf_time} which is less than opts in the block {remaining_time}")