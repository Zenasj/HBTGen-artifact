device_model = getattr(torch, device_type, None)
if device_model:
    num_gpus_per_host = getattr(device_model, "device_count")()