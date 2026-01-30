should_cast_forward_inputs = len(state._handles) > 0 and all(
    not handle._force_full_precision for handle in state._handles
)