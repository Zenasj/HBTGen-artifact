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

# Computing key_averages() changes what is returned!
prof.key_averages()
remaining_again, rf_again = sum(e.self_cpu_time_total for e in remaining), rf_event.self_cpu_time_total
print(f"Other events before: {remaining_time} vs after {remaining_again}")
print(f"rf event before: {rf_time} vs after {rf_again}")