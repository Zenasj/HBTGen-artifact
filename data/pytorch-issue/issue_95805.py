for handle in handles:
        flat_param = handle.flat_param
        already_registered = hasattr(flat_param, "_post_backward_hook_state")
        if already_registered or not flat_param.requires_grad:
            continue
        ...
        hook_handle = acc_grad.register_hook(
            functools.partial(_post_backward_hook, state, handle)
        )