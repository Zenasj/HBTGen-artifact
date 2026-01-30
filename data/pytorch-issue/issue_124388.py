for fe in function_events:
            if (
                fe.device_type == DeviceType.CPU
                and not fe.is_async
                and fe.id in device_corr_map
            ):
                for f_evt in device_corr_map[fe.id]:
                    if f_evt.device_type == DeviceType.CUDA:
                        fe.append_kernel(
                            f_evt.name,
                            f_evt.device_index,
                            f_evt.time_range.end - f_evt.time_range.start,
                        )