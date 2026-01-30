# root instance will have no direct flat param handles
assert state._is_root
state._handles  # []
# decision will return `True` even in `model.eval()` mode
should_cast_forward_inputs = all(
    not handle._force_full_precision for handle in state._handles
)
should_cast_forward_inputs  # `True`

should_cast_forward_inputs = len(state._handles) > 0 and all(
    not handle._force_full_precision for handle in state._handles
)