import torch

def catch_errors_wrapper(callback, hooks: Hooks):
    @functools.wraps(callback)
    def catch_errors(frame, cache_size, frame_state):
        assert frame_state is not None
        
        if (
            # TODO: the first condition is not covered by any test
            frame.f_lasti >= first_real_inst_idx(frame.f_code)
            or skipfiles.check(frame.f_code.co_filename)
            or config.disable
        ):  
            log.debug("skipping %s %s", frame.f_code.co_name, frame.f_code.co_filename)
            return None
        if frame.f_code.co_filename == "<string>" and frame.f_code.co_name == "__new__":
            # nametuple constructor
            return None
        if config.optimize_ddp:
            ddp_module = DistributedDataParallel._get_active_ddp_module()
            if ddp_module:
                with compile_lock:
                    from torch._dynamo.backends.distributed import DDPOptimizer
                    
                    ddp_optimizer = DDPOptimizer(
                        bucket_bytes_cap=ddp_module.bucket_bytes_cap,
                        backend_compile_fn=callback._torchdynamo_orig_callable,
                    )
                    hijacked_callback = convert_frame.convert_frame(
                        ddp_optimizer.compile_fn,
                        hooks=hooks,
                    )
                    return hijacked_callback(frame, cache_size, hooks, frame_state)
        
        with compile_lock, _disable_current_modes():
            return callback(frame, cache_size, hooks, frame_state)
    
    catch_errors._torchdynamo_orig_callable = callback  # type: ignore[attr-defined]
    return catch_errors