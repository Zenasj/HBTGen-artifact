with FSDP.state_dict_type(
                    model, StateDictType.LOCAL_STATE_DICT, LocalStateDictConfig(offload_to_cpu=True)
            ):
                fsdp_sd = model.state_dict()

with FSDP.state_dict_type(
                    model, StateDictType.LOCAL_STATE_DICT, LocalStateDictConfig(offload_to_cpu=False)
            ):
                fsdp_sd = model.state_dict()