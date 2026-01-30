import torch

def udf_with_torch_ops(device=-1, use_record_function=False):
    device_ctx = contextlib.suppress() if device == -1 else torch.cuda.device(device)
    record_function_ctx = (
        torch.autograd.profiler.record_function("##forward##")
        if use_record_function
        else contextlib.suppress()
    )
    with device_ctx, record_function_ctx:
        t1, t2 = torch.ones(1), torch.ones(1)
        t = torch.add(t1, t2)
        t = torch.mul(t, t)
        t = t.relu()
        t = t.sigmoid()

def test_udf_with_torch_ops(self):
        with torch.autograd.profiler.profile() as prof:
            udf_with_torch_ops(use_record_function=True)
        function_events = prof.function_events
        record_function_event = [
            evt for evt in function_events if "##forward##" in evt.name
        ][0]
        remaining_events = {
            evt for evt in function_events
        } - {record_function_event}
        # These ops are created by the hack of casting record_function to a
        # tensor, so they should not count in the actual UDF profiled time.
        # TODO remove after https://github.com/pytorch/pytorch/issues/43868
        # is resolved.
        events_denylist = [
            "aten::zeros",
            "aten::empty",
            "aten::zero_",
            "aten::fill_",
        ]
        # Time of all ops under record_function ctx manager.
        ops_time = sum(
            evt.cpu_time_total
            for evt in remaining_events
            if not any(
                [
                    rf_entry_event in evt.name
                    for rf_entry_event in events_denylist
                ]
            )
        )
        if record_function_event.cpu_time_total < ops_time:
            print(prof.key_averages().table())
            print("--- GOING TO FAIL ---")
        self.assertGreaterEqual(
            record_function_event.cpu_time_total, ops_time
        )
        print(prof.key_averages().table())