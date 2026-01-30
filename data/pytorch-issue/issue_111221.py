dynamo_guarded_config_ignorelist = {
    "log_file_name",
    "verbose",
    "verify_correctness",  # will not affect model, will raise RuntimeError
    # (no silent change to compilation behaviour)
    "cache_size_limit",
    "accumulated_cache_size_limit",
    "print_specializations",
    "replay_record_enabled",
    "cprofile",  # only wraps _compile, not graph
    "repro_after",
    "repro_level",
    "repro_forward_only",
    "repro_tolerance",
    "same_two_models_use_fp64",
    "error_on_recompile",  # safe because: will throw error
    "report_guard_failures",
    "report_all_guard_failures",
    "base_dir",  # used for minifying / logging
    "translation_validation",
    "translation_validation_timeout",
    "translation_validation_no_bisect",
    "DEBUG_DIR_VAR_NAME",
    "debug_dir_root",
}