state_dict = module.state_dict()
module.load_state_dict(state_dict, strict=True)