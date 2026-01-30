def to_mkldnn(state_dict):
    state_dict_new = {}
    for k,p in state_dict.items():
        if not 'fpn' in k:
            try:
                state_dict_new[k] = p.to_mkldnn()
            except Exception as exp:
                print(f'{k} caused : {exp.args}.')
        # else:
        #     print(f'\t fpn skipped :{k}')
    return state_dict_new