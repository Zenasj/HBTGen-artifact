with FSDP.FullyShardedDataParallel.state_dict_type(
        trainer.model,
        StateDictType.LOCAL_STATE_DICT, # or any other StateDictType
        LocalStateDictConfig(offload_to_cpu=True), # or without this line
        LocalOptimStateDictConfig(offload_to_cpu=True), # or without this line
        ):
    state_dict = trainer.model.state_dict()

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    with FSDP.FullyShardedDataParallel.state_dict_type(
            trainer.model,
            StateDictType.LOCAL_STATE_DICT, # or any other StateDictType
            LocalStateDictConfig(offload_to_cpu=True), # or without this line
            LocalOptimStateDictConfig(offload_to_cpu=True), # or without this line
            ):
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa