def _register_jit_decomposition_for_jvp(decomp, use_python=False):
    if decomp in decomposition_table_for_jvp:
        decomposition_table_used = decomposition_table_for_jvp
    elif decomp in decomposition_table:
        decomposition_table_used = decomposition_table
    else:
        raise RuntimeError(f"could not find decomposition for {decomp}")
    decomp_fn = decomposition_table_used[decomp]

    # `out_wrapper` extends a decompositions signature with
    # an `out` parameter. However jit will use the unwrapped function's
    # signature instead so we need to unwrap here to prevent an error
    decomp_fn = _maybe_remove_out_wrapper(decomp_fn)
    ...