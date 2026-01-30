self._train_for_several_steps(
    fsdp_model,
    num_steps=1,
    autocast=False,
    mixed_precision=fsdp_kwargs["mixed_precision"],
)