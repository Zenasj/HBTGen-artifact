def function(inputs):
    inputs_ref_0 = inputs[1]
    graph_out_1 = __compiled_fn_2(inputs)
    getitem_1 = graph_out_1[0]
    add = inputs_ref_0
    del graph_out_1
    add_1 = add + getitem_1
    add = None
    getitem_1 = None
    cpu = add_1.cpu()
    add_1 = None
    return (cpu,)