dist_cp.load_state_dict(
    state_dict=state_dict,
    storage_reader=storage_reader,
    planner=None,
    process_group=process_group,
)

state_dict = model.state_dict()
dcp_cp.load_state_dict(state_dict, ...)
model.load_state_dict(state_dict)