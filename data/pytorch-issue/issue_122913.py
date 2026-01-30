@record_shapeenv_event()
def _constrain_symbol_range(shape_env, s: sympy.Symbol, compiler_min: int, compiler_max: int):
    upd_vr = ValueRanges(compiler_min, compiler_max)
    old_vr = shape_env.var_to_range.get(s, ValueRanges.unknown())
    shape_env._update_var_to_range(s, upd_vr)
    if (new_vr := shape_env.var_to_range[s]) != old_vr:
        log.info("_constrain_symbol_range %s [%s, %s]", s, new_vr.lower, new_vr.upper)