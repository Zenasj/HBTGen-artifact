import torch

def _custom_setup_context(
    setup_context_fn=None,
    *,
    device_type: str,
    cast_inputs: Optional[torch.dtype] = None,
):
    """The missing amp setup_context decorator for custom ops."""

    if setup_context_fn is None:
        return functools.partial(
            _custom_setup_context, device_type=device_type, cast_inputs=cast_inputs
        )

    @functools.wraps(setup_context_fn)
    def decorate_setup_context(ctx, *args, **kwargs):
        ctx._dtype = torch.get_autocast_dtype(device_type)
        if cast_inputs is None:
            ctx._fwd_used_autocast = torch.is_autocast_enabled(device_type)
            return setup_context_fn(ctx, *args, **kwargs)
        else:
            autocast_context = torch.is_autocast_enabled(device_type)
            ctx._fwd_used_autocast = False
            if autocast_context:
                with torch.autocast(device_type=device_type, enabled=False):
                    return setup_context_fn(
                        ctx,
                        *torch.amp.autocast_mode._cast(args, device_type, cast_inputs),
                        **torch.amp.autocast_mode._cast(kwargs, device_type, cast_inputs),
                    )
            else:
                return setup_context_fn(ctx, *args, **kwargs)

    return decorate_setup_context