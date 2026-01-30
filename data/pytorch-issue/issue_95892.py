with FSDP.state_dict_type(
    model,
    StateDictType.SHARDED_STATE_DICT,
):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': FSDP.optim_state_dict(model, optimizer)
    }

    save_state_dict(state_dict, FileSystemWriter(ckpt_path))

with FSDP.state_dict_type(
    model,
    StateDictType.SHARDED_STATE_DICT,
):
    storage_reader = FileSystemReader(ckpt_path)
    state_dict = {
        'model': model.state_dict(),
    }
    load_state_dict(state_dict, storage_reader)
    state_dict |= load_sharded_optimizer_state_dict(
        model_state_dict=state_dict['model'],
        optimizer_key='optimizer',
        storage_reader=storage_reader,
    )