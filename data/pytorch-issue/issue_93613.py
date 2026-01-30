import torch

def _convert_node_to_placeholder(node, inps_dict):
    if node.op == 'output' or node.op == "placeholder":
        return
    node.op = 'placeholder'
    node.args = ()
    node.kwargs = {}
    node.target = node.name
    concrete_val = node.meta.get('concrete_value', None)
    if isinstance(concrete_val, torch.Tensor):
        inps_dict[node.name] = concrete_val
    else:
        inps_dict[node.name] = torch.zeros(())
        for tuple_user in list(node.users):
            _convert_node_to_placeholder(tuple_user, inps_dict)

@register_strategy("Delta Debugging")
def delta_debugging(cur_graph: fx.Graph, cur_inps, granularity):
    num_nodes = len(cur_graph.nodes)
    for start_range in range(0, num_nodes, granularity):
        is_removing = False
        new_graph = deepcopy_fx_graph(cur_graph)
        #new_inps = cur_inps[:]
        ph_names = [node.name for node in get_placeholders(cur_graph)]
        new_inps_dict = dict(zip(ph_names, cur_inps[:]))
        end_range = min(num_nodes, start_range + granularity)
        for idx in range(start_range, end_range):
            new_node = list(new_graph.nodes)[idx]
            if new_node.op not in ['placeholder', 'output']:
                is_removing = True
                _convert_node_to_placeholder(new_node, new_inps_dict)
        if not is_removing:
            continue
        new_graph = _consolidate_placeholders(new_graph)
        ph_names = [node.name for node in get_placeholders(new_graph)]
        new_inps = [new_inps_dict[name] for name in ph_names]
        new_state = remove_unused_inputs_unchecked(ReproState(new_graph, new_inps))
        if new_state is None:
            new_state = ReproState(new_graph, new_inps)
        if graph_fails(new_state.graph, new_state.inps):
            return ReproState(new_state.graph, new_state.inps)

    return None