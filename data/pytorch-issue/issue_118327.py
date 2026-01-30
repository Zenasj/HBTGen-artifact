state_dict_list = [state_dict['state']]
dist.broadcast_object_list(state_dict_list, src=0)
state_dict['state'] = state_dict_list[0]