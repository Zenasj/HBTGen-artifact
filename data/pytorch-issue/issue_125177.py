set_optimizer_state_dict(
                    model=self.model,
                    optimizers=optimizer,
                    optim_state_dict=optim_state_dict,
                    options=StateDictOptions(
                        full_state_dict=self.fsdp_state_dict_type == 'full',
                        strict=strict,
                        cpu_offload=True,
                    ),
                )

optim_state_dict = MagicMock() if optim_state_dict is None else optim_state_dict