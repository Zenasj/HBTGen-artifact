est_timing: List[Tuple[triton.runtime.Config, float]]
est_timing = [
   (config, perf_model(**named_args, **kwargs, **config.all_kwargs()))
    for config in configs
]
configs = sorted(est_timing, key=lambda x: est_timing[1])[:top_k]

def call_prune_configs(  # type: ignore[no-untyped-def]
    autotuner,
    early_config_prune: Callable,
    perf_model: Callable,
    top_k: float,
    is_top_k_float: bool,
    configs: List,
    named_args: Dict,
    kwargs: Dict,
):
    if early_config_prune:
        configs = early_config_prune(configs, named_args, **kwargs)

    if perf_model:
        # we assert top_k is a float before calling this
        if is_top_k_float and top_k <= 1.0:
            top_k = int(len(configs) * top_k)
        if len(configs) > top_k:
            est_timing = [
                (config, perf_model(**named_args, **kwargs, **config.all_kwargs()))
                for config in configs
            ]
            configs = sorted(est_timing, key=lambda x: est_timing[1])[:top_k]
    return configs
    
    
# Called in torch/_higher_order_ops/triton_kernel_wrap.py
pruned_configs = self.call_user_defined_fn(
    call_prune_configs,
    [
        variable,
        wrapped_early_configs_prune,
        wrapped_perf_model,
        wrapped_configs_top_k,
        wrapped_is_top_k_float,
        wrapped_configs,
        named_args,
        kwargs,
    ],
    {},
    tx,
    variable.source,
)