device_kernel_cache = Path.home() / ".cache" / "torch" / "aoti_eager" / name_space.lower() / device_type.lower()
op_overload_name = op_overload_name if op_overload_name else "default"
op_kernel_cache_config = device_kernel_cache / f"{op_func_name}.{op_overload_name}.json"